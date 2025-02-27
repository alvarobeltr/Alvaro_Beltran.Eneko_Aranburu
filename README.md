# 📡 Zirkuitu Lineal eta Ez-Linealen Proiektua

## 🔍 Proiektuaren Deskribapena
Proiektu honek zirkuitu elektriko lineal eta ez-linealen analisia egiten du. **Matrizeen bidezko analisian** oinarritzen da, eta `numpy` liburutegia erabiltzen du zirkuituen **matrize-eredua** sortzeko eta ebazteko.

## 📂 Fitxategiak
- `main.py` → Programa nagusia.
- `circuit.txt` → Zirkuituaren elementuak deskribatzen dituen fitxategia.
- `analyzer.py` → Matriz bidezko analisia egiten duen modulu nagusia.
- `README.md` → Dokumentazio hau.

## 🚀 Erabilera
1. **Instalatu beharrezko liburutegiak:**
   ```bash
   pip install numpy
   ```
2. **Exekutatu programa:**
   ```bash
   python main.py circuit.txt
   ```
3. **Emaitzak ikusiko dituzu terminalean edo emaitza fitxategian.**

## 🔧 Funtzionamendua
- **Zirkuituaren irakurketa:** `numpy.loadtxt()` erabiliz.
- **Nodoen eta adarren analisia:** `np.unique()` eta `np.where()` erabiliz.
- **Matrizeen eraikuntza:** Erabiltzen dira matrize **inzidentzia**, **eroankortasun**, eta **iturburu bektoreak**.
- **Ebazpena:** `numpy.linalg.solve()` erabiliz.

## 📌 Adibidezko Datuak
```
V_1 1 0 0 0 10 0 0 0
R_1 1 2 0 0 1e3 0 0 0
R_2 2 3 0 0 2e3 0 0 0
```

## 📜 Lizentzia
MIT Lizentzia.

## 🤝 Egileak
- **Alvaro Beltrán de Nanclares**
- **Eneko Aranburu**
