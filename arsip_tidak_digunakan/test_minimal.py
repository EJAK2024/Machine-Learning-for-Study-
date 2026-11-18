from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 10, "Hello World test PDF")
pdf.output("test_minimal.pdf")
print("Minimal PDF created")
