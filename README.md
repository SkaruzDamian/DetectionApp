# Detection App

System analizy chodu i postawy z nagrań wideo.

## O projekcie

Ta aplikacja powstała do analizy techniki biegania i chodzenia. Używa bibliotek MediaPipe i OpenCV do automatycznej detekcji kluczowych parametrów:

- **Sposób stawiania stopy** - czy lądujemy na pięcie, śródstopiu czy palcach
- **Pochylenie ciała** - kąt nachylenia tułowia i głowy podczas ruchu

Program przeanalizuje każdą klatkę filmu i wygeneruje szczegółowy raport z procentowym rozkładem technik oraz wykresami.

## Wymagania

- Python 3.8+
- Windows 10/11
- Pliki wideo w formacie MP4

## Szybki start

### 1. Pobierz i zainstaluj

Pobierz repozytorium z Githuba, a następnie zainstaluj zależności z pliku requirements.txt
```bash
pip install -r requirements.txt
```

### 2. Dodaj filmy do analizy

Wrzuć pliki MP4 do folderu `videos/`.

### 3. Uruchom analizę

**Podstawowa wersja** (szybsza, mniej dokładna):
```bash
python app1.py
```

**Zaawansowana wersja** (wolniejsza, bardziej dokładna):
```bash
python app2.py
```

Program pokazuje postęp na bieżąco.

## Co dostaniesz w wyniku

Po analizie w folderze `output1/` lub `output2/` znajdziesz:

### Film z adnotacjami
- `nazwa_filmu_analyzed.mp4` - oryginalny film z nałożonymi wskaźnikami w czasie rzeczywistym

### Dane liczbowe  
- `nazwa_filmu_analysis.csv` - dane z każdej klatki (timestamp, typ uderzenia, kąty)
- `nazwa_filmu_report.json` - podsumowanie statystyczne

### Wizualizacje
- `nazwa_filmu_visualization.png` - wykresy kołowe i przebiegi czasowe

## Różnice między wersjami

**Wersja 1** - do szybkich testów
- Prosty algorytm detekcji
- Analiza około 2x szybsza
- Gorsze radzenie sobie z trudnymi warunkami (słabe światło, ruch kamery)

**Wersja 2** - do dokładnych analiz
- Kombinuje kilka metod detekcji (geometria + prędkość + historia)
- Wygładza dane, interpoluje brakujące pomiary  
- Lepiej radzi sobie z niedoskonałymi nagraniami


## Zaawansowane

### Dostosowanie parametrów

W kodzie możesz zmienić progi detekcji:

```python
# W app2.py
HEEL_THRESHOLD = 12      # Czułość detekcji pięty
TOE_THRESHOLD = 12       # Czułość detekcji palców  
VELOCITY_LOW_THRESHOLD = 0.01  # Próg prędkości dla klasyfikacji
```

### MediaPipe landmarks

Program używa tych punktów:
- **Stopy:** kostki (27,28), pięty (29,30), palce (31,32)
- **Postura:** nos (0), uszy (7,8), ramiona (11,12), biodra (23,24)

### Format danych wyjściowych

CSV zawiera kolumny:
- `frame_number` - numer klatki
- `timestamp` - czas w sekundach
- `left_foot_strike` / `right_foot_strike` - typ uderzenia (heel/midfoot/forefoot)
- `forward_lean` / `head_forward` - kąty w stopniach

## Zależności

```
opencv-python==4.8.1.78
mediapipe==0.10.7  
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
typing-extensions>=4.0.0
```
