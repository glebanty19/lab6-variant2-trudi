"""Format LR4/LR6 reports in the style of the supplied university example."""

from __future__ import annotations

import re
from collections import defaultdict

from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph


CODE_URL = "https://github.com/glebanty19/lab6-variant2-trudi"

GOALS = {6: 'оцінити вплив зміни деталізації на детекцію контейнерів TRUDI, виміряти доменний зсув, '
    'порівняти масштабну аугментацію й донавчання та перевірити плитковий інференс із контекстом.'}

THEMES = {6: 'Дослідження доменного зсуву та адаптації детектора контейнерів на аерознімках (варіант 2, '
    'TRUDI)'}

TASKS = {6: {1: 'Сформулювати задачу, перевірити TRUDI й зафіксувати протокол оцінювання.',
     2: 'Виміряти зміну якості й ознак детектора за відносних масштабів ×1, ×2 та ×4.',
     3: 'Порівняти масштабну аугментацію, контрольне навчання й донавчання на цільовій '
        'деталізації.',
     4: 'Перевірити плитковий інференс, перекриття та просторовий контекст.',
     5: 'Узагальнити протокол, результати та обмеження експерименту.',
     6: 'Описати використання ШІ й відповісти на контрольні питання.'}}

TABLE_TITLES = {6: ['Склад підмножин TRUDI',
     'Навчання та контрольні метрики',
     'Якість за відносного масштабу',
     'Ознаки доменного зсуву',
     'Порівняння методів адаптації',
     'Якість методів на test ×4',
     'Використання цільових даних',
     'Плитки з контекстом',
     'Паспорт експерименту']}


def _remove_paragraph(paragraph: Paragraph) -> None:
    paragraph._p.getparent().remove(paragraph._p)


def _set_runs(paragraph: Paragraph, size: int = 14, bold: bool | None = None) -> None:
    for run in paragraph.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0, 0, 0)
        if bold is not None:
            run.bold = bold


def _cover_line(anchor: Paragraph, text: str, *, size: int = 14,
                bold: bool = False, italic: bool = False,
                after: float = 0, before: float = 0,
                left: float = 0, heading: bool = False) -> Paragraph:
    paragraph = anchor.insert_paragraph_before(style="Heading 1" if heading else "Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if left else WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.left_indent = Cm(left)
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.0
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = bold
    run.italic = italic
    return paragraph


def _page_number(section) -> None:
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = ""
    section.first_page_footer.paragraphs[0].text = ""
    section.footer.paragraphs[0].text = ""
    paragraph = section.header.paragraphs[0]
    paragraph.text = ""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    for kind, value in (("begin", None), (None, "PAGE"), ("separate", None),
                        (None, "1"), ("end", None)):
        run = paragraph.add_run()
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        if kind:
            field = OxmlElement("w:fldChar")
            field.set(qn("w:fldCharType"), kind)
            run._r.append(field)
        elif value == "PAGE":
            instruction = OxmlElement("w:instrText")
            instruction.text = "PAGE"
            run._r.append(instruction)
        else:
            run.text = value
    section_properties = section._sectPr
    numbering = section_properties.find(qn("w:pgNumType"))
    if numbering is None:
        numbering = OxmlElement("w:pgNumType")
        section_properties.append(numbering)
    numbering.set(qn("w:start"), "0")


def _style_table(table: Table) -> None:
    table.style = "Table Grid"
    table.autofit = True
    properties = table._tbl.tblPr
    for old in properties.findall(qn("w:tblBorders")):
        properties.remove(old)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "777777")
        borders.append(border)
    properties.append(borders)
    columns = len(table.columns)
    longest = max(len(cell.text) for row in table.rows for cell in row.cells)
    size = 14 if columns <= 4 else 12 if columns == 5 else 10
    if columns >= 4 and longest > 28:
        size = min(size, 10)
    if columns == 4 and "bbox" in table.cell(0, 1).text:
        widths = [Cm(2.7), Cm(4.8), Cm(4.8), Cm(5.1)]
        table.autofit = False
        for index, width in enumerate(widths):
            table.columns[index].width = width
            for row in table.rows:
                row.cells[index].width = width
    for row_index, row in enumerate(table.rows):
        if row_index == 0:
            row_properties = row._tr.get_or_add_trPr()
            if row_properties.find(qn("w:tblHeader")) is None:
                row_properties.append(OxmlElement("w:tblHeader"))
        for cell in row.cells:
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for shading in cell._tc.get_or_add_tcPr().findall(qn("w:shd")):
                cell._tc.get_or_add_tcPr().remove(shading)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.first_line_indent = Cm(0)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                paragraph.alignment = (WD_ALIGN_PARAGRAPH.CENTER if row_index == 0
                                       or paragraph.text.strip().replace(",", "").replace(".", "").isdigit()
                                       else WD_ALIGN_PARAGRAPH.LEFT)
                _set_runs(paragraph, size, bold=True if row_index == 0 else None)


def style_report(doc, lab_number: int) -> None:
    """Restyle an existing generated report without rebuilding tables or images."""
    if lab_number != 6:
        raise ValueError("Only laboratory work 6 is supported")

    section = doc.sections[0]
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)
    section.top_margin = Cm(2.10)
    section.bottom_margin = Cm(2.08)
    section.left_margin = Cm(3.00)
    section.right_margin = Cm(1.15)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)
    _page_number(section)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(14)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    heading = doc.styles["Heading 1"]
    heading.font.name = "Times New Roman"
    heading.font.size = Pt(14)
    heading.font.bold = True
    heading.font.color.rgb = RGBColor(0, 0, 0)
    heading.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    heading.paragraph_format.space_before = Pt(18)
    heading.paragraph_format.space_after = Pt(6)
    heading.paragraph_format.keep_with_next = True

    task_headings = {}
    source_heading = None
    for paragraph in list(doc.paragraphs):
        if paragraph.style.name != "Heading 1":
            continue
        title = paragraph.text.strip()
        if title == "Джерела":
            source_heading = paragraph
            continue
        match = re.match(r"^(\d+)\.", title)
        if not match:
            continue
        original = int(match.group(1))
        if original == 2:
            _remove_paragraph(paragraph)
            continue
        if lab_number == 4:
            task = {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}.get(original)
            if original in (8, 9):
                _remove_paragraph(paragraph)
                continue
            if original == 10:
                paragraph.text = "Висновок"
                continue
        else:
            task = {1: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}.get(original)
            if original == 8:
                _remove_paragraph(paragraph)
                continue
        if task:
            paragraph.text = f"Завдання {task}"
            task_headings[task] = paragraph

    if set(task_headings) != set(range(1, 7)):
        raise ValueError(f"Missing task sections: {set(range(1, 7)) - set(task_headings)}")
    if source_heading is None:
        raise ValueError("Source section is missing")

    if lab_number == 6:
        conclusion = next((p for p in doc.paragraphs if p.text.startswith("Висновок:")), None)
        if conclusion is None:
            raise ValueError("LR6 conclusion paragraph is missing")
        source_heading._p.addprevious(conclusion._p)
        title = conclusion.insert_paragraph_before("Висновок", style="Heading 1")
        title.paragraph_format.keep_with_next = True

    old_cover = list(doc.paragraphs[:4])
    anchor = old_cover[0]
    cover = [
        _cover_line(anchor, "Хмельницький національний університет", after=17.35),
        _cover_line(anchor, "Факультет інформаційних технологій", after=17.35),
        _cover_line(anchor, "Кафедра комп’ютерних наук", after=45),
        _cover_line(anchor, f"ЛАБОРАТОРНА РОБОТА №{lab_number}", size=16, bold=True, after=18.45),
        _cover_line(anchor, "з дисципліни «Штучний інтелект для циркулярної економіки»", size=16, heading=True),
        _cover_line(anchor, f"на тему «{THEMES[lab_number]}»", size=16, heading=True),
        _cover_line(anchor, "Виконав:", left=10, after=5),
        _cover_line(anchor, "студент групи КНм-26-1", left=10, italic=True, after=5),
        _cover_line(anchor, "Бондар Глєб Ігорович", left=10, after=15),
        _cover_line(anchor, "Хмельницький 2026", before=225),
    ]
    cover_elements = {id(p._p) for p in cover}
    for paragraph in old_cover:
        _remove_paragraph(paragraph)

    code_link = next((p for p in doc.paragraphs if CODE_URL in p.text), None)
    goal_anchor = code_link or task_headings[1]
    goal = goal_anchor.insert_paragraph_before()
    goal.add_run("Мета: ").bold = True
    goal.add_run(GOALS[lab_number])
    goal.paragraph_format.page_break_before = True
    goal_element = id(goal._p)

    intro_elements = set()
    for number, task_heading in task_headings.items():
        intro = doc.add_paragraph(TASKS[lab_number][number])
        task_heading._p.addnext(intro._p)
        intro.paragraph_format.keep_with_next = True
        intro_elements.add(id(intro._p))

    task_counter = defaultdict(int)
    current_task = 0
    table_index = 0
    for element in list(doc.element.body.iterchildren()):
        if element.tag == qn("w:p"):
            paragraph = Paragraph(element, doc)
            match = re.fullmatch(r"Завдання (\d+)", paragraph.text.strip())
            if match:
                current_task = int(match.group(1))
        elif element.tag == qn("w:tbl"):
            if table_index >= len(TABLE_TITLES[lab_number]):
                raise ValueError("More tables than expected")
            table_index += 1
            task_counter[current_task] += 1
            caption = doc.add_paragraph(
                f"Таблиця {current_task}.{task_counter[current_task]} – "
                f"{TABLE_TITLES[lab_number][table_index - 1]}"
            )
            element.addprevious(caption._p)
            _style_table(Table(element, doc))
    if table_index != len(TABLE_TITLES[lab_number]):
        raise ValueError(f"Expected {len(TABLE_TITLES[lab_number])} tables, got {table_index}")

    for paragraph in doc.paragraphs:
        if id(paragraph._p) in cover_elements:
            continue
        fmt = paragraph.paragraph_format
        text = paragraph.text.strip()
        if paragraph.style.name == "Heading 1":
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.first_line_indent = Cm(0)
            fmt.space_before = Pt(18)
            fmt.space_after = Pt(6)
            fmt.keep_with_next = True
            _set_runs(paragraph, 14, bold=True)
        elif id(paragraph._p) == goal_element:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt.first_line_indent = Cm(1)
            fmt.line_spacing = 1.25
            fmt.space_after = Pt(10)
            _set_runs(paragraph)
        elif text.startswith("Таблиця "):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.first_line_indent = Cm(0)
            fmt.space_before = Pt(6)
            fmt.space_after = Pt(5)
            fmt.keep_with_next = True
            if lab_number == 4 and text.startswith("Таблиця 1.2"):
                fmt.page_break_before = True
            _set_runs(paragraph)
        elif paragraph._p.xpath(".//w:drawing"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.first_line_indent = Cm(0)
            fmt.space_before = Pt(6)
            fmt.space_after = Pt(0)
        elif text.startswith("Рисунок "):
            paragraph.text = paragraph.text.replace(" — ", " – ")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.first_line_indent = Cm(0)
            fmt.space_after = Pt(10)
            _set_runs(paragraph)
        elif CODE_URL in text:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            fmt.first_line_indent = Cm(0)
            fmt.line_spacing = 1.0
            fmt.space_after = Pt(10)
            _set_runs(paragraph, 12)
        else:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            fmt.first_line_indent = Cm(1.49)
            fmt.line_spacing = 1.25
            fmt.space_after = Pt(6)
            if id(paragraph._p) in intro_elements:
                fmt.keep_with_next = True
            _set_runs(paragraph)

    doc.core_properties.title = f"Лабораторна робота №{lab_number}, варіант 2"
