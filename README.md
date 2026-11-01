# programi za zavrsni rad: detekcija zgrada na ortofotou novog sada

programi napisani za zavrsni (bachelor) rad na fakultetu tehnickih nauka u novom sadu
(autor: pal piri, mentor: prof. miro govedarica).

dva modela u-net sa enkoderom resnet34 trenirana su na javnim skupovima podataka
(whu building dataset i inria aerial image labeling, gradovi bec i zapadni tirol),
primenjena na ortofoto novog sada rezolucije 30 cm, ocenjena na tri rucno oznacena
podrucja (petrovaradin, kampus, centar), dotrenirana postupkom "izostavi jedno
podrucje" i na kraju pokrenuta u qgis-u preko dodatka deepness.

## redosled pokretanja

| korak | program | sta radi | izlaz |
| --- | --- | --- | --- |
| 1 | provera_prevzorkovanja.py | (u python konzoli qgis-a) proverava kojim nacinom je ortofoto smanjen sa 10 cm na 30 cm | tabela u konzoli |
| 2 | tile_inria.py | sece snimke skupa inria (bec, zapadni tirol) na isecke 512 x 512, snimci 1-5 idu u validaciju, oznake 0/1, izbacuje 80 % praznih isecaka | isecci za treniranje i validaciju |
| 3 | train.py | trenira u-net/resnet34 (dice + bce, adam 1e-4, 20 epoha) i cuva model sa najvecim validacionim iou; pokrece se dva puta, za whu i za inria (promeni se data_root i ime izlazne datoteke) | best_model_whu.pth, best_model_inria.pth |
| 4 | podaci_treniranja.py | broji isecke, ponovo racuna validacioni iou sacuvanih modela i meri brzinu treniranja | brojevi za rad |
| 5 | broj_parametara.py | broji parametre mreze (enkoder, dekoder, izlazni sloj) | brojevi za rad |
| 6 | predict.py | prva verzija predvidjanja: prozori bez preklapanja (vide se savovi) | karta zgrada |
| 7 | predict_overlap.py | predvidjanje na celom ortofotou sa preklapanjem 50 % i hann tezinama; pokrece se za oba modela | campus_buildings_whu_overlap.tif, campus_buildings_inria.tif |
| 8 | pravi_isecke.py | rasterizuje rucno nacrtane zgrade i podrucja na mrezu ortofotoa od 30 cm | maska_30cm.tif, oblasti_30cm.tif |
| 9 | za_proveru.py | pravi spisak mesta gde su rucne oznake mozda pogresne (oba modela se ne slazu sa oznakom) | za_proveru.gpkg |
| 10 | brojanje_zgrada.py | broji zgrade po podrucjima (po centroidu) i racuna udeo zgrada | brojevi za tabelu 4.2 |
| 11 | ocena.py | mere po pikselima (iou, preciznost, odziv, f1) i po objektima, po podrucjima; karte gresaka | rezultati_ocena.csv, greske_whu.tif, greske_inria.tif |
| 12 | slika_savovi.py | poredi predvidjanje bez preklapanja i sa preklapanjem i crta sliku 4.4 | slika_savovi_whu.png, slika_savovi_inria.png |
| 13 | dotreniravanje.py | sest dotreniravanja (2 modela x 3 podrucja) po postupku "izostavi jedno podrucje", ocena pre i posle | rezultati_dotreniravanje.csv, dotreniran_*.tif, dotreniran_*_bez_*.pth |
| 14 | izvoz_onnx.py | cuva dotrenirani model inria (bez kampusa) u formatu onnx sa podesavanjima za deepness i proverava ga | deepness_zgrade_inria.onnx |
| 15 | ocena_deepness.py | poredi rezultat iz deepness-a sa rucnim oznakama i sa predvidjanjem iz pythona | tabela u konzoli |

posle pregleda oznaka (korak 9) ispravljene oznake se ponovo pretvaraju u masku
(korak 8), pa se ponovo pokrecu koraci 10 i 11.

ocena_deepness.py, dotreniravanje.py i slika_savovi.py uvoze funkcije iz ocena.py,
pa sva cetiri programa moraju biti u istom folderu.

## podesavanja i putanje

svaki program na pocetku ima deo "podesavanja" sa putanjama i parametrima. putanje
su apsolutne i odgovaraju racunaru na kome je rad uradjen (folder c:\thesis, podaci u
c:\thesis\data); na drugom racunaru treba ih promeniti. ispisi programa su na engleskom.

ocekivani raspored podataka:

```
c:\thesis\data\whu\{train,val,test}\{image,label}\*.tif      skup whu (vec isecen na 512 x 512)
c:\thesis\data\inria\train\{images,gt}\*.tif                 skup inria (original)
c:\thesis\data\inria_tiles\...                               pravi tile_inria.py
c:\thesis\data\campus_30cm.tif                               ortofoto novog sada, 30 cm
c:\thesis\data\kampus_oznake.gpkg                            rucne oznake (slojevi zgrade i oblasti)
c:\thesis\data\isecci_kampus\...                             pravi pravi_isecke.py
```

## sta nije u repozitorijumu

- skupovi whu i inria (preuzimaju se sa sajtova autora);
- ortofoto novog sada i rucne oznake;
- sacuvani modeli (.pth i .onnx, oko 98 mb svaki) i rasteri sa rezultatima.

## instalacija

```
conda create -n thesis python
conda activate thesis
pip install -r requirements.txt
```

pytorch sa podrskom za graficku karticu (cuda) instalira se prema uputstvu sa pytorch.org.
treniranje je radjeno na grafickoj kartici nvidia gtx 1050 ti (4 gb).
provera_prevzorkovanja.py se pokrece u qgis-u, jer treba gdal koji cita ecw datoteke.
