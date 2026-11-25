"""Add the public LR6 code link to the generated Word report."""

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Pt


CODE_URL = "https://github.com/glebanty19/lab6-variant2-trudi"


def add_code_link(doc: Document) -> None:
    paragraph = doc.add_paragraph()
    paragraph.add_run("Код ЛР6 на GitHub: ")
    relationship = doc.part.relate_to(CODE_URL, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    run.append(properties)
    text = OxmlElement("w:t")
    text.text = CODE_URL
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
    paragraph.paragraph_format.space_after = Pt(10)
