# Programi za završni rad: detekcija zgrada na ortofotou Novog Sada

Ovde su programi napisani za završni (bachelor) rad na Fakultetu tehničkih nauka u Novom Sadu.

- Autor: Pal Piri
- Mentor: prof. dr Miro Govedarica

## Šta je urađeno

1. Mreža U-Net sa enkoderom ResNet34 trenirana je dva puta: na skupu WHU i na delu skupa Inria (Beč i Zapadni Tirol).
2. Oba modela su primenjena na ortofoto Novog Sada, rezolucije 30 cm.
3. Na ortofotou su ručno nacrtane zgrade na tri područja: Petrovaradin, kampus i centar. Na tim područjima su modeli ocenjeni.
4. Modeli su zatim dotrenirani na ručno nacrtanim zgradama. Svaki put je model dotreniran na dva područja, a ocenjen na trećem.
5. Na kraju je dotrenirani model Inria pokrenut u programu QGIS, pomoću dodatka Deepness.

## Redosled pokretanja

| Korak | Program | Šta radi | Rezultat |
| --- | --- | --- | --- |
| 1 | provera_prevzorkovanja.py | Proverava kojom metodom je ortofoto smanjen sa 10 cm na 30 cm. Pokreće se u Python konzoli QGIS-a. | Tabela u konzoli |
| 2 | tile_inria.py | Seče snimke skupa Inria (Beč, Zapadni Tirol) na isečke 512 × 512 piksela. Snimci 1–5 idu u validaciju. Oznake pretvara u 0 i 1. Izbacuje 80 % isečaka bez zgrada. | Isečci za treniranje i validaciju |
| 3 | train.py | Trenira U-Net sa enkoderom ResNet34 (Dice + BCE, Adam 1e-4, 20 epoha). Čuva model sa najvećim validacionim IoU. Pokreće se dva puta, za WHU i za Inria. Pre drugog pokretanja promene se `DATA_ROOT` i `MODEL_OUT`. | best_model_whu.pth, best_model_inria.pth |
| 4 | podaci_treniranja.py | Broji isečke, ponovo računa validacioni IoU sačuvanih modela i meri brzinu treniranja. | Brojevi za rad |
| 5 | broj_parametara.py | Broji parametre mreže: u enkoderu, u dekoderu i u poslednjem sloju. | Brojevi za rad |
| 6 | predict.py | Prva verzija predviđanja. Prozori se ne preklapaju, pa se na rezultatu vide šavovi. | Karta zgrada |
| 7 | predict_overlap.py | Predviđanje na celom ortofotou. Prozori se preklapaju 50 % i spajaju se sa Hann težinama. Pokreće se za oba modela. | campus_buildings_whu_overlap.tif, campus_buildings_inria.tif |
| 8 | pravi_isecke.py | Pretvara ručno nacrtane zgrade i područja u raster na mreži ortofotoa od 30 cm. Seče isečke 512 × 512 piksela. | maska_30cm.tif, oblasti_30cm.tif, isečci |
| 9 | za_proveru.py | Pravi spisak mesta gde su ručne oznake možda pogrešne, tj. gde se oba modela ne slažu sa oznakom. | za_proveru.gpkg |
| 10 | brojanje_zgrada.py | Broji zgrade po područjima (po centroidu). Računa koliki deo područja zauzimaju zgrade. | Brojevi za tabelu 4.2 |
| 11 | ocena.py | Računa mere po pikselima (IoU, preciznost, odziv, F1) i po objektima, za svako područje. Pravi karte grešaka. | rezultati_ocena.csv, greske_whu.tif, greske_inria.tif |
| 12 | slika_savovi.py | Poredi predviđanje bez preklapanja i sa preklapanjem. Crta sliku 4.4. | slika_savovi_whu.png, slika_savovi_inria.png |
| 13 | dotreniravanje.py | Šest dotreniravanja (2 modela × 3 područja). Svaki put se jedno područje izostavi i na njemu se model oceni. Ocenjuje modele pre i posle dotreniravanja. | rezultati_dotreniravanje.csv, dotreniran_*.tif, dotreniran_*_bez_*.pth |
| 14 | izvoz_onnx.py | Čuva dotrenirani model Inria (dotreniran bez kampusa) u formatu ONNX, sa podešavanjima za Deepness. Zatim ga proverava. | deepness_zgrade_inria.onnx |
| 15 | ocena_deepness.py | Poredi rezultat iz Deepness-a sa ručnim oznakama i sa predviđanjem iz Python-a. | Tabela u konzoli |
| 16 | napravi_average_30cm.py | Dodatna provera: smanjuje ortofoto sa 10 cm na 30 cm metodom average, iz punog snimka od 10 cm. Poredi ga sa campus_30cm.tif. Pokreće se u Python konzoli QGIS-a. Posle njega se predict_overlap.py i ocena.py pokreću na novom snimku. | campus_30cm_average.tif, tabela u konzoli |

Napomene:

- Posle pregleda oznaka (korak 9) ispravljene oznake se ponovo pretvaraju u masku (korak 8). Zatim se ponovo pokreću koraci 10 i 11.
- Svi programi moraju biti u istom folderu kao podesavanja.py.
- Programi ocena_deepness.py, dotreniravanje.py i slika_savovi.py koriste funkcije iz ocena.py. Zato i ocena.py mora biti u istom folderu.

## Podešavanja i putanje

Sve putanje su u jednoj datoteci, podesavanja.py. Tu su dve vrednosti:

- `THESIS_DIR` – folder sa modelima (u radu C:\thesis),
- `DATA` – folder sa podacima (u radu C:\thesis\data).

Na drugom računaru promeni se samo `THESIS_DIR`, a ako su podaci na drugom mestu, i `DATA`. Ostala podešavanja (parametri) su na početku svakog programa. Programi ispisuju poruke na engleskom.

Raspored podataka:

```
C:\thesis\data\whu\{train,val,test}\{image,label}\*.tif    skup WHU (već isečen na 512 x 512)
C:\thesis\data\inria\train\{images,gt}\*.tif               skup Inria (originalni snimci)
C:\thesis\data\inria_tiles\...                             pravi ga tile_inria.py
C:\thesis\data\campus_30cm.tif                             ortofoto Novog Sada, 30 cm
C:\thesis\data\kampus_oznake.gpkg                          ručne oznake (slojevi zgrade i oblasti)
C:\thesis\data\isecci_kampus\...                           pravi ga pravi_isecke.py
C:\thesis\data\deepness_zgrada.shp                         rezultat dodatka Deepness (za ocena_deepness.py)
```

## Ponovljivost

- Programi train.py i dotreniravanje.py koriste seed 42, pa novo pokretanje pravi iste nasumične izbore.
- Seed je u train.py dodat posle izrade rada. Polazni modeli iz rada trenirani su bez seed-a, pa novo treniranje ne daje potpuno iste modele.
- Na grafičkoj kartici neki proračuni nisu potpuno isti pri svakom pokretanju, pa rezultat i sa seed-om može malo da se razlikuje.

## Šta nije u repozitorijumu

- Skupovi WHU i Inria. Preuzimaju se sa sajtova autora.
- Ortofoto Novog Sada i ručne oznake.
- Sačuvani modeli (.pth i .onnx, oko 98 MB svaki) i rasteri sa rezultatima. Preveliki su za GitHub.

## Instalacija

```
conda create -n thesis python=3.11
conda activate thesis
pip install -r requirements.txt
```

- Verzije biblioteka u requirements.txt su iste kao u radu (tabela 4.1). PyTorch sa podrškom za grafičku karticu (CUDA 12.6) preuzima se sa sajta pytorch.org.
- Treniranje je rađeno na grafičkoj kartici NVIDIA GTX 1050 Ti (4 GB).
- Programi provera_prevzorkovanja.py i napravi_average_30cm.py se pokreću u QGIS-u (3.34.2), jer koriste GDAL iz QGIS-a, koji čita ECW datoteke.

## Licenca

MIT. Vidi datoteku LICENSE.
