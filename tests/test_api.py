import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from uuid import uuid4
import json

from backend.main import app
from backend.database.connection import init_db_pool, close_db_pool

# Use function-scoped fixture to match pytest-asyncio's event loop scope
@pytest_asyncio.fixture(scope="function")
async def db_pool():
    pool = await init_db_pool()
    yield pool
    await close_db_pool()

def make_pdf(text: str) -> bytes:
    """
    Generates a valid PDF with correct cross-reference table offsets for any given text,
    including standard Helvetica font resources so text extractors can read it.
    """
    content = f"BT\n/F1 12 Tf\n72 712 Td\n({text}) Tj\nET"
    content_bytes = content.encode("ascii")
    
    obj1 = b"1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    obj2 = b"2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n"
    
    # Page object with font resource definition
    obj3 = (
        b"3 0 obj\n"
        b"<<\n"
        b"/Type /Page\n"
        b"/Parent 2 0 R\n"
        b"/MediaBox [0 0 612 792]\n"
        b"/Resources <<\n"
        b"  /Font <<\n"
        b"    /F1 <<\n"
        b"      /Type /Font\n"
        b"      /Subtype /Type1\n"
        b"      /BaseFont /Helvetica\n"
        b"    >>\n"
        b"  >>\n"
        b">>\n"
        b"/Contents 4 0 R\n"
        b">>\n"
        b"endobj\n"
    )
    
    obj4_header = f"4 0 obj\n<<\n/Length {len(content_bytes)}\n>>\nstream\n".encode("ascii")
    obj4_footer = b"\nendstream\nendobj\n"
    obj4 = obj4_header + content_bytes + obj4_footer
    
    pdf_header = b"%PDF-1.4\n"
    
    offset1 = len(pdf_header)
    offset2 = offset1 + len(obj1)
    offset3 = offset2 + len(obj2)
    offset4 = offset3 + len(obj3)
    
    xref_offset = offset4 + len(obj4)
    
    xref = (
        "xref\n"
        "0 5\n"
        "0000000000 65535 f\n"
        f"{offset1:010d} 00000 n\n"
        f"{offset2:010d} 00000 n\n"
        f"{offset3:010d} 00000 n\n"
        f"{offset4:010d} 00000 n\n"
    ).encode("ascii")
    
    trailer = (
        "trailer\n"
        "<<\n"
        "/Size 5\n"
        "/Root 1 0 R\n"
        ">>\n"
        "startxref\n"
        f"{xref_offset}\n"
        "%%EOF\n"
    ).encode("ascii")
    
    return pdf_header + obj1 + obj2 + obj3 + obj4 + xref + trailer

@pytest.mark.asyncio
async def test_health_check(db_pool):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_qa_endpoint(db_pool):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Fetch the entire stream finite response directly
        response = await ac.get("/api/qa", params={
            "query": "Can an IFSC Banking Unit accept deposits from Indian residents?"
        }, timeout=45.0)
        
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        lines = response.text.split("\n")
        
        # Verify we got done signal
        done = [l for l in lines if l.startswith("event: done")]
        assert len(done) > 0
        
        # Verify we got tokens
        tokens = [l for l in lines if l.startswith("event: token")]
        assert len(tokens) > 0

@pytest.mark.asyncio
async def test_admin_list_documents(db_pool):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/admin/documents")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

@pytest.mark.asyncio
async def test_admin_stats(db_pool):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/api/admin/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "doc_count" in stats
        assert "node_count" in stats

@pytest.mark.asyncio
async def test_admin_ingest_and_logs(db_pool):
    # Dynamic valid PDF with text window large enough for classification (minimum 50 chars)
    pdf_data = make_pdf("The International Financial Services Centres Authority Act, 2019 is an Act to provide for the establishment of an Authority to regulate financial services.")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Ingest document
        response = await ac.post(
            "/api/admin/ingest",
            files={"file": ("test_ingest.pdf", pdf_data, "application/pdf")}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"
        
        # Connect to logs SSE with follow=false
        log_response = await ac.get("/api/admin/ingest/logs", params={"follow": "false"})
        assert log_response.status_code == 200
        assert "text/event-stream" in log_response.headers["content-type"]
        
        lines = log_response.text.split("\n")
        assert len(lines) > 0
        
        # Ensure done event or log event exists
        log_events = [l for l in lines if l.startswith("event: log") or l.startswith("event: done")]
        assert len(log_events) > 0

@pytest.mark.asyncio
async def test_compliance_endpoint(db_pool):
    # Dynamic valid PDF representing compliance practices text
    pdf_data = make_pdf("We are an IFSC Banking Unit accepting deposits of USD 5,000 from retail customers.")
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Run compliance audit
        response = await ac.post(
            "/api/compliance",
            files={"file": ("entity_practices.pdf", pdf_data, "application/pdf")},
            timeout=45.0
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        
        lines = response.text.split("\n")
        done = [l for l in lines if l.startswith("event: done")]
        assert len(done) > 0
