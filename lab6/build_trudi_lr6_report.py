#!/usr/bin/env python3
"""Build LR6 variant 2 report from the actual TRUDI experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

from report_utils import add_code_link
from sample_report_style import style_report

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"runs/trudi_report_artifacts"
SOURCE="trudi_lr4_source"
CONTROL="trudi_lr4_continue_native"
AUG="trudi_lr4_scale_aug"
ADAPT="trudi_lr4_adapt_x4"


def load(path):
    return json.loads((ROOT/path).read_text(encoding="utf8"))


def metric(model,run):
    return load(f"runs/{model}/{run}/{run}_metrics.json")


def fmt(value,digits=3):
    return (f"{value:.{digits}f}" if value is not None else "—").replace(".",",")


def table(doc,headers,rows):
    t=doc.add_table(rows=1,cols=len(headers));t.style="Table Grid"
    for c,value in zip(t.rows[0].cells,headers):c.text=str(value)
    for row in rows:
        for c,value in zip(t.add_row().cells,row):c.text=str(value)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:r.font.name="Arial";r.font.size=Pt(8)
    doc.add_paragraph()


def figure(doc,path,caption,width=6.2):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(ROOT/path),width=Inches(width))
    p=doc.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.italic=True;r.font.size=Pt(9)


def charts(source,control,aug,adapt):
    OUT.mkdir(parents=True,exist_ok=True)
    factors=[1,1.5,2,3,4]
    vals=[source[f]["mAP50_95"] for f in factors]
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(factors,vals,marker="o",label="Native-trained source")
    ax.plot(factors,[control[f]["mAP50_95"] for f in factors],marker="d",label="Continued native training")
    ax.plot(factors,[aug[f]["mAP50_95"] for f in factors],marker="s",label="Scale augmentation")
    ax.plot([1,2,4],[adapt[f]["mAP50_95"] for f in (1,2,4)],marker="^",label="Fine-tuning at ×4")
    ax.set(xlabel="Simulated pixel-size multiplier",ylabel="mAP@[.5:.95]",title="TRUDI aerial container detection")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/"transfer.png",dpi=170);plt.close(fig)
    histories=[(load(f"runs/{SOURCE}/history.json"),"Native model"),
               (load(f"runs/{CONTROL}/history.json"),"Continued native"),
               (load(f"runs/{AUG}/history.json"),"Scale augmentation")]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for rows,name in histories:
        axes[0].plot([r["epoch"] for r in rows],[r["train_loss"] for r in rows],marker="o",label=name)
        axes[1].plot([r["epoch"] for r in rows],[r["val_map50_95"] for r in rows],marker="o",label=name)
    axes[0].set(xlabel="Epoch",ylabel="Train loss");axes[1].set(xlabel="Epoch",ylabel="Validation mAP@[.5:.95]")
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(OUT/"training.png",dpi=170);plt.close(fig)
    tile=metric(SOURCE,"test_tiles")
    context=metric(SOURCE,"test_tiles_context")
    larger_context=metric(SOURCE,"test_tiles_800_context")
    fig,ax=plt.subplots(figsize=(8.5,3.8))
    names=["Whole frame","Tile 640","Tile 640 + context 128","Tile 800 + context 128"]
    ax.bar(names,[source[1]["mAP50_95"],tile["mAP50_95"],context["mAP50_95"],larger_context["mAP50_95"]])
    ax.set(ylabel="mAP@[.5:.95]",title="Spatial context on the same test images")
    ax.tick_params(axis="x",labelrotation=22);fig.tight_layout();fig.savefig(OUT/"context.png",dpi=170);plt.close(fig)


def main():
    import torch
    source_checkpoint=torch.load(ROOT/"runs"/SOURCE/"best.pt",map_location="cpu",weights_only=False)
    initial_weights=source_checkpoint.get("source_checkpoint")
    selected=load("runs/variant2_audit/selected_lr4_model.json")
    if Path(initial_weights or "").as_posix()!=Path(selected["checkpoint"]).as_posix():
        raise ValueError("LR6 source was not initialized from the selected LR4 checkpoint")
    audit=load("runs/variant2_audit/dataset_audit.json")["lr6"]
    shift=load("runs/variant2_audit/lr6_feature_shift.json")
    source={1:metric(SOURCE,"test_native"),1.5:metric(SOURCE,"test_x1_5"),
            2:metric(SOURCE,"test_x2"),3:metric(SOURCE,"test_x3"),
            4:metric(SOURCE,"test_x4")}
    control={1:metric(CONTROL,"test_native"),1.5:metric(CONTROL,"test_x1_5"),
             2:metric(CONTROL,"test_x2"),3:metric(CONTROL,"test_x3"),
             4:metric(CONTROL,"test_x4")}
    aug={1:metric(AUG,"test_native"),1.5:metric(AUG,"test_x1_5"),
         2:metric(AUG,"test_x2"),3:metric(AUG,"test_x3"),
         4:metric(AUG,"test_x4")}
    adapt={1:metric(ADAPT,"test_native"),2:metric(ADAPT,"test_x2"),
           4:metric(ADAPT,"test_x4")}
    native_val=metric(SOURCE,"val")
    control_val=metric(CONTROL,"val")
    aug_val=metric(AUG,"val")
    adapt_val=metric(ADAPT,"val_x4")
    tiles=metric(SOURCE,"test_tiles")
    context=metric(SOURCE,"test_tiles_context")
    larger_context=metric(SOURCE,"test_tiles_800_context")
    thresholds=(native_val["best_f1_threshold"]["confidence"],aug_val["best_f1_threshold"]["confidence"],
                adapt_val["best_f1_threshold"]["confidence"])
    charts(source,control,aug,adapt)
    doc=Document();sec=doc.sections[0]
    sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.7)
    sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
    doc.styles["Normal"].font.name="Arial";doc.styles["Normal"].font.size=Pt(10)
    for name in ("Title","Heading 1","Heading 2"):
        doc.styles[name].font.name="Arial";doc.styles[name].font.color.rgb=RGBColor(31,78,121)
    sec.header.paragraphs[0].text="Лабораторна робота № 6 · Доменний зсув"
    sec.footer.paragraphs[0].text="Варіант 2 · TRUDI aerial"
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(90)
    r=p.add_run("ЛАБОРАТОРНА РОБОТА № 6");r.bold=True;r.font.size=Pt(22)
    p=doc.add_paragraph("Дослідження доменного зсуву та адаптації\nВаріант 2")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.font.size=Pt(16)
    p=doc.add_paragraph("Дисципліна: Штучний інтелект для циркулярної економіки\nВиконав: Бондар Глєб Ігорович, КНм-26-1\nХмельницький, 2026")
    p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()
    add_code_link(doc)

    doc.add_heading("1. Постановка та відповідність варіанту",level=1)
    doc.add_paragraph("За методичкою варіант 2: детектор контейнерів ЛР4, вихідний GSD 3 см, емульовані 6 і 12 см, масштабна аугментація, донавчання, плитки з контекстом. ЛР4 варіанта 2 задає купи будівельних відходів, тому її голову для контейнерів безпосередньо використати не можна. Для предметної частини ЛР6 використано аерознімки контейнерів TRUDI. Джерельну TRUDI модель ініціалізовано сумісними шарами фінальної моделі ЛР4 на DroneWaste; класифікаційну голову для контейнерів створено заново й навчено на TRUDI train.")
    doc.add_paragraph("TRUDI містить переважно вантажні контейнери на портових і транспортних майданчиках, танк-контейнери й трейлери; їх не слід називати спеціальними контейнерами для відходів. У цьому досліді container і tank_container об’єднано в один позитивний клас «контейнер»; trailer залишається фоном. Фізичного GSD для цих кадрів у локальних COCO анотаціях немає. Тому ×1, ×2, ×4 є відносними рівнями деталізації, а числа 3/6/12 см не виміряні.")

    doc.add_heading("2. Дані, перевірка та протокол",level=1)
    table(doc,["Підмножина","Кадри","Контейнери + tank","Трейлери","Усі об’єкти"],
          [[split,str(audit["splits"][split]["images"]),
            str(audit["splits"][split]["classes"].get("container",0)+audit["splits"][split]["classes"].get("tank_container",0)),
            str(audit["splits"][split]["classes"].get("trailer",0)),str(audit["splits"][split]["annotations"])]
           for split in ("train","val","test")])
    test_objects=audit["splits"]["test"]["classes"].get("container",0)+audit["splits"]["test"]["classes"].get("tank_container",0)
    doc.add_paragraph(f"Перевірено наявність кожного зображення, відповідність класів, додатність і межі COCO рамок. Однакових назв файлів між трьома підмножинами немає. Цільові зображення для оцінки отримано лише з відкладеного test, тому вони не перетинаються з train. Вихідна і деградована оцінки використовують попарно ті самі 75 test сцен, щоб ізолювати вплив масштабу. Це перевірка формального перетину; кадри одного порту можуть бути візуально подібними, тому незалежність локацій не доведено. На test {test_objects} позитивних об’єктів.")
    figure(doc,"runs/variant2_audit/trudi_18_boxes.png","Рисунок 1 — Перевірка 18 аерознімків із рамками, по шість на підмножину.",6.5)
    doc.add_paragraph(f"Протокол: вихідну, продовжену контрольну й аугментовану моделі вибрано за validation mAP@[0.5:0.95] при ×1, а донавчену для цілі — за validation ×4. Поріг confidence кожної моделі підібрано за micro-F1 на відповідній validation підмножині та зафіксовано для всіх її test режимів: базова {fmt(thresholds[0])}; продовжена {fmt(control_val['best_f1_threshold']['confidence'])}; аугментація {fmt(thresholds[1])}; донавчання ×4 — {fmt(thresholds[2])}. На test не обирали ані поріг, ані checkpoint. AP обчислюється власною 101-точковою COCO-подібною реалізацією, без повної сумісності з pycocotools. Об’єднання плиток — NMS з IoU 0,5.")
    table(doc,["Модель","Train loss, остання","Краща епоха","Val mAP","Val F1 threshold"],
          [[name,fmt(rows[-1]["train_loss"]),str(max(rows,key=lambda r:r["val_map50_95"])["epoch"]),fmt(val["mAP50_95"]),fmt(val["best_f1_threshold"]["confidence"])]
           for name,rows,val in (("Базова TRUDI",load(f"runs/{SOURCE}/history.json"),native_val),
                                 ("Продовження без аугментації",load(f"runs/{CONTROL}/history.json"),control_val),
                                 ("Масштабна аугментація",load(f"runs/{AUG}/history.json"),aug_val),
                                 ("Донавчання ×4",load(f"runs/{ADAPT}/history.json"),adapt_val))])
    figure(doc,"runs/trudi_report_artifacts/training.png","Рисунок 2 — Криві навчання та валідації.")

    doc.add_heading("3. Вимірювання масштабу й доменного зсуву",level=1)
    base=source[1]["mAP50_95"]
    table(doc,["Фактор","mAP@0.5","mAP@[.5:.95]","Δ mAP","Відносна Δ"],
          [[f"×{factor:g}",fmt(source[factor]["AP50"]),fmt(source[factor]["mAP50_95"]),
            fmt(source[factor]["mAP50_95"]-base),f"{100*(source[factor]['mAP50_95']/base-1):+.1f}%"]
           for factor in (1,1.5,2,3,4)])
    doc.add_paragraph(f"Єдиний цільовий клас — контейнери (container + tank_container), GT на test: {test_objects}; тому AP цього класу дорівнює наведеному mAP. AP@0,5={fmt(source[1]['AP50'])} на вихідній деталізації і {fmt(source[4]['AP50'])} на ×4.")
    doc.add_paragraph("Сцену зменшено фільтром BOX і повернуто білінійною інтерполяцією до початкової геометрії; GT рамки лишилися ті самі. Отже, змінюється лише розподіл зображень X, а склад об’єктів і мітки Y фіксовані. Значення за ×1,5 і ×3 додано для вимог завдання 3. Невеликі немонотонні коливання mAP не слід тлумачити як покращення фізичного GSD.")
    figure(doc,f"runs/{SOURCE}/test_native/test_native_pr.png","Рисунок 3 — PR-крива базової моделі на test.")
    stats=shift["channel_statistics"]
    table(doc,["Фактор","R mean","G mean","B mean","R std","G std","B std","Відстань ознак L2"],
          [["×"+factor,*[fmt(v,4) for v in stats[factor]["mean"]],*[fmt(v,4) for v in stats[factor]["std"]],
            fmt(shift["feature_mean_distance"][factor]["l2"],4)] for factor in ("1","2","4")])
    doc.add_paragraph(f"RGB гістограми та ROI ознаки fc7 Faster R-CNN обчислено на тих самих перших {shift['images']} test кадрах. Для гістограм кожен кадр зменшено до 256×256 і нормовано до [0,1]. Для ознак усереднено пропозиції в межах кожного кадру, потім кадри; наведено евклідову відстань між середніми ознаками вихідного та цільового масштабів. Це описовий аналіз, а не перевірка статистичної значущості.")
    doc.add_paragraph(f"Відстань середніх ROI ознак від ×1 становить {fmt(shift['feature_mean_distance']['2']['l2'],4)} для ×2 і {fmt(shift['feature_mean_distance']['4']['l2'],4)} для ×4. За незмінних GT це штучно створений коваріатний зсув зображень і ознак; зсуву міток між попарними оцінками немає. Дані не доводять наявності концептуального зсуву або переносимості на інший сенсор.")
    figure(doc,"runs/variant2_audit/lr6_rgb_histograms.png","Рисунок 4 — Гістограми яскравості RGB для ×1, ×2, ×4.")

    doc.add_heading("4. Два способи зменшення зсуву",level=1)
    doc.add_paragraph("Контроль: вихідну TRUDI модель додатково навчено 4 епохи без зміни масштабу. Спосіб 1: той самий початковий checkpoint донавчено 4 епохи з випадковим масштабом 1, 2 або 4 для кожного train кадру. Спосіб 2: початковий checkpoint донавчено 3 епохи на розміченому train із постійним фактором ×4; його вибрано за validation ×4. Усі способи не використовують test для навчання.")
    table(doc,["Метод","Джерело ×1","Ціль ×1,5","Ціль ×2","Ціль ×3","Ціль ×4"],
          [["Без адаптації",*[fmt(source[f]["mAP50_95"]) for f in (1,1.5,2,3,4)]],
           ["Продовження без аугментації",*[fmt(control[f]["mAP50_95"]) for f in (1,1.5,2,3,4)]],
           ["Масштабна аугментація",fmt(aug[1]["mAP50_95"]),fmt(aug[1.5]["mAP50_95"]),fmt(aug[2]["mAP50_95"]),fmt(aug[3]["mAP50_95"]),fmt(aug[4]["mAP50_95"])],
           ["Донавчання ×4",fmt(adapt[1]["mAP50_95"]),"—",fmt(adapt[2]["mAP50_95"]),"—",fmt(adapt[4]["mAP50_95"]) ]])
    table(doc,["Метод на test ×4","Поріг","AP@0,5","mAP@[.5:.95]","Precision","Recall","F1"],
          [[title,fmt(row["threshold"]),fmt(row["AP50"]),fmt(row["mAP50_95"]),
            fmt(row["operating_point"]["precision"]),fmt(row["operating_point"]["recall"]),
            fmt(row["operating_point"]["f1"])]
           for title,row in (("Без адаптації",source[4]),("Контроль",control[4]),
                             ("Аугментація",aug[4]),("Донавчання ×4",adapt[4]))])
    table(doc,["Спосіб","Нові цільові сцени","Розмічені train сцени","Де використана цільова деталізація"],
          [["Без адаптації","0","231","Лише оцінювання"],
           ["Продовження без аугментації","0","231","Лише оцінювання"],
           ["Масштабна аугментація","0","231","Випадково ×1/×2/×4 у train"],
           ["Донавчання ×4","0","231","Усі train кадри ресемпльовано ×4"]])
    figure(doc,"runs/trudi_report_artifacts/transfer.png","Рисунок 5 — Матриця переносимості за масштабом.")
    aug_delta=aug[4]["mAP50_95"]-source[4]["mAP50_95"]
    change="підвищила" if aug_delta>=0 else "знизила"
    controlled_delta=aug[4]["mAP50_95"]-control[4]["mAP50_95"]
    controlled_change="краща" if controlled_delta>=0 else "гірша"
    doc.add_paragraph(f"Відносно вихідної моделі аугментація {change} mAP на ×4 на {fmt(abs(aug_delta))}. За однакової кількості додаткових епох вона {controlled_change} за продовження без аугментації на {fmt(abs(controlled_delta))} mAP ×4; це точніше відокремлює вплив кількості епох від масштабу. Один seed і одна тестова підмножина не дають підстав вважати малу різницю статистично значущою. Донавчання ×4 порівнюється на ×1, ×2 і ×4. Зміна метрик не підтверджує перенос на інший порт, сенсор або фізично виміряний GSD.")

    doc.add_heading("5. Плитки та контекст",level=1)
    table(doc,["Режим","mAP@0.5","mAP@[.5:.95]","Precision","Recall",f"F1 за порога {fmt(thresholds[0])}"],
          [[title,fmt(row["AP50"]),fmt(row["mAP50_95"]),
            fmt(row.get("operating_point",{}).get("precision")),
            fmt(row.get("operating_point",{}).get("recall")),
            fmt(row.get("operating_point",{}).get("f1"))]
           for title,row in (("Цілий кадр",source[1]),("Плитка 640 px, overlap 20%",tiles),
                             ("Плитка 640 + контекст 128 px",context),
                             ("Плитка 800 + контекст 128 px",larger_context))])
    doc.add_paragraph("Плитковий режим розбиває велике аерозображення на ядра 640 або 800 px із перекриттям 20%. Контекстний режим розширює кожне ядро на 128 px, виконує детекцію на ширшому полі та залишає рамки з центром у ядрі. Потім рамки повертаються в координати повного кадру й зливаються NMS. Оцінюються однакові 75 test кадрів і ті самі GT; різниця між 640 і 800 px показує вплив розміру плитки за сталого контексту.")
    tile_change="зріс" if larger_context["mAP50_95"]>=context["mAP50_95"] else "зменшився"
    doc.add_paragraph(f"За контексту 128 px mAP@[.5:.95] {tile_change} з {fmt(context['mAP50_95'])} до {fmt(larger_context['mAP50_95'])} при переході від ядра 640 до 800 px. Це результат для наявних розмірів кадрів; кращий розмір на інших сценах потрібно перевіряти окремо.")
    figure(doc,"runs/trudi_report_artifacts/context.png","Рисунок 6 — Повний кадр, плитки двох розмірів і контекст.")

    doc.add_heading("6. Паспорт експерименту та висновки",level=1)
    table(doc,["Параметр","Значення"],[["Дані","TRUDI aerial COCO; train/val/test 231/77/75"],
          ["Клас","container + tank_container; trailer — фон"],["Модель","Faster R-CNN ResNet-50 FPN; голова 2 класи"],
          ["Ініціалізація",f"Сумісні шари {initial_weights}; нова голова для TRUDI"],
          ["Навчання",f"seed 42, SGD lr 0,001, batch 1, {len(load(f'runs/{SOURCE}/history.json'))} епох базової моделі"],
          ["Масштаб","BOX downsample ×1/1,5/2/3/4 + bilinear upsample; реальний GSD невідомий"],
          ["Вибір","validation mAP checkpoint; validation micro-F1 threshold"],
          ["Метрики","101-точкова COCO-подібна AP, IoU 0,5:0,95; test 75 кадрів"],
          ["Обладнання","NVIDIA RTX 4060 Laptop GPU; PyTorch 2.11.0+cu128"],
          ["Артефакти","run_trudi_lr6.py, trudi_lr6.py, analyze_lr6.py, runs/trudi_lr4_*/"],
          ["Контроль навчання","4 додаткові епохи від того самого джерельного checkpoint без масштабної аугментації"]])
    doc.add_paragraph("Рекомендації: для строгого варіанта 2 зібрати аерознімки контейнерів із записаним GSD та незалежними портами, перевірити повноту розмітки, розділити сцени за локаціями/датами, оцінити масштаб на кількох незалежних доменах. Наявні результати показують вплив ресемплінгу та двох способів адаптації лише для TRUDI.")
    context_change="підвищив" if context["mAP50_95"]>=tiles["mAP50_95"] else "знизив"
    frame_comparison="повний кадр залишився кращим" if source[1]["mAP50_95"]>=max(context["mAP50_95"],larger_context["mAP50_95"]) else "найкращі контекстні плитки перевищили повний кадр"
    doc.add_paragraph(f"Висновок: на 75 test аерознімках базова модель отримала mAP@[0.5:0.95]={fmt(source[1]['mAP50_95'])} при ×1 та {fmt(source[4]['mAP50_95'])} при ×4. Масштабна аугментація дала {fmt(aug[4]['mAP50_95'])} на ×4, донавчання — {fmt(adapt[4]['mAP50_95'])}. Контекст {context_change} плиткове mAP із {fmt(tiles['mAP50_95'])} до {fmt(context['mAP50_95'])}; плитка 800 px із контекстом дала {fmt(larger_context['mAP50_95'])}; {frame_comparison}. Невідомий фізичний GSD і один відкладений split обмежують висновки про інші умови знімання.")
    doc.add_heading("7. Використання ШІ",level=1)
    doc.add_paragraph("ChatGPT/Codex застосовано для аналізу умови, перевірки відповідності датасетів, доповнення коду аудиту та оцінювання і підготовки структури звіту. Приклади запитів: «Як емулювати зміну GSD без зміни bbox?», «Як добрати поріг без витоку test?», «Як виміряти зсув ROI ознак?». Рекомендації перевірено локальним запуском checkpoint, аудитом COCO та зіставленням JSON із таблицями. Власний висновок: ресемплінг показує відносний зсув, але не доводить перенос на реальні GSD 3/6/12 см. Автору необхідно самостійно переглянути код і пояснити вибір метрик та обмеження GSD на захисті.")
    doc.add_heading("8. Контрольні питання",level=1)
    answers=[
        ("Що таке об’єктний масштаб?","Розмір цільового об’єкта в пікселях; він залежить від фізичного розміру, GSD, оптики та висоти. FPN і масштабна аугментація допомагають працювати з різними масштабами."),
        ("Навіщо потрібне перекриття плиток?","Щоб об’єкти на межах частіше цілком потрапляли хоча б в одну плитку. За кроку s=tile·(1−overlap) приблизна кількість плиток — (ceil((W−tile)/s)+1)·(ceil((H−tile)/s)+1); крайні плитки додають до меж кадру. Перекриття обирають за розміром об’єкта та ціною дубльованих обчислень."),
        ("Як GSD та світло впливають на модель?",f"Більший GSD зменшує число деталей у пікселях; висота знімання змінює масштаб, а освітлення — контраст, тіні й колір. У нашому тесті mAP змінився з {fmt(source[1]['mAP50_95'])} при ×1 до {fmt(source[4]['mAP50_95'])} при ×4; фізичний GSD не виміряно."),
        ("Що таке просторовий контекст?","Оточення об’єкта, яке допомагає відрізнити його від схожого фону; у роботі це додаткові 128 px навколо плитки."),
        ("Чому потрібне суміщення сенсорів?","Багатосенсорне/мультимодальне об’єднання поєднує різні спостереження тієї самої сцени після калібрування, часової та геометричної прив’язки, узгодження роздільності й обробки пропусків. Без суміщення канали описують різні точки й раннє об’єднання стає некоректним."),
        ("Які є схеми об’єднання?","Рання схема з’єднує суміщені канали на вході й доречна за точного вирівнювання; проміжна об’єднує навчені ознаки, коли модальності різні; пізня усереднює/зважує рішення окремих моделей і терпить слабше суміщення. У демонстраційному прикладі методички рання схема могла використати локальні взаємодії вирівняних каналів; у цій роботі мультимодального набору немає."),
        ("Які види доменного зсуву?","Коваріатний змінює розподіл зображень, зсув міток — частоти класів, концептуальний — залежність мітки від ознак."),
        ("Як виміряти зсув без розмітки?","Порівняти гістограми, середні й коваріації ознак або навчити класифікатор доменів. Якість детекції без GT не виміряти."),
        ("Як працюють нормалізація, histogram matching та AdaBN?","Нормалізація вирівнює середнє й дисперсію каналів і потребує лише цільового зображення. Histogram matching наближає його гістограму до еталона джерела, тож потребує еталонних кадрів. AdaBN перераховує статистики BatchNorm на нерозмічених цільових кадрах; у Faster R-CNN із FrozenBatchNorm пряме AdaBN без зміни архітектури непридатне."),
        ("Чим адаптація відрізняється від узагальнення?","Адаптація використовує цільові дані, інколи розмічені, щоб підлаштувати модель; узагальнення готує модель лише на джерелах до невідомого домену. CORAL штрафує різницю коваріацій ознак, DANN через обернення градієнта вчить ознаки, за якими дискримінатор не розрізняє домени."),
        ("Чому фотометрична аугментація інколи не допомагає?","Вона не відновлює втраченої просторової деталізації. Аугментацію слід узгоджувати з очікуваним типом зсуву."),
        ("Що таке leave-one-domain-out?","По черзі вилучати один домен для тесту, навчати на решті й подавати матрицю якісних переходів та розкид метрик. Відкладений домен не використовують для добору параметрів, якщо явно не оголошено адаптацію; її обсяг розмітки записують у паспорті разом із джерелом, ціллю, розділенням, GSD та межами застосовності."),
    ]
    for q,a in answers:
        p=doc.add_paragraph();p.add_run(q).bold=True;doc.add_paragraph(a)
    doc.add_heading("Джерела",level=1)
    doc.add_paragraph("Методичні вказівки: «Лабораторна робота № 04» і «Лабораторна робота № 06», локальні PDF.")
    doc.add_paragraph("TRUDI and TITUS: Gülsoylu et al., BMVC 2025, https://github.com/egulsoylu/trudi")
    doc.add_paragraph("DroneWaste v1.0: https://zenodo.org/records/17045559")
    style_report(doc, 6)
    output=ROOT.parent/"Звіт_Лаб6_Варіант2.docx"
    try:
        doc.save(output)
    except PermissionError:
        output=ROOT.parent/"Звіт_Лаб6_Варіант2_для_здачі.docx"
        doc.save(output)
    print(f"Saved {output}")


if __name__=="__main__":main()
