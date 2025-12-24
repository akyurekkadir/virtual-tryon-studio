# 🚀 GitHub'a Yükleme Rehberi

## Adım 1: GitHub'da Repo Oluştur

1. GitHub'a git: https://github.com/new
2. Repository name: `virtual-tryon-studio` (veya istediğin isim)
3. Description: "AI-powered virtual try-on application with ComfyUI"
4. Public veya Private seç
5. **❌ Initialize with README seçme** (bizde zaten var)
6. "Create repository" butonuna tıkla

## Adım 2: Terminal Komutları

Projenin bulunduğu klasöre git ve şu komutları çalıştır:

```bash
# Git repository'sini başlat
git init

# Tüm dosyaları ekle
git add .

# İlk commit
git commit -m "Initial commit: Virtual Try-On Studio with Color Harmony Analysis"

# Ana branch'i main yap
git branch -M main

# GitHub repo'nuzu bağlayın (URL'i kendi repo'nuzla değiştirin!)
git remote add origin https://github.com/KULLANICI_ADIN/virtual-tryon-studio.git

# GitHub'a yükle
git push -u origin main
```

## Adım 3: GitHub Token (Şifre Yerine)

Eğer şifre soruyorsa:

1. GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. "Generate new token (classic)"
3. Scope: `repo` seç
4. Token'ı kopyala
5. Terminal'de şifre yerine bu token'ı kullan

## ✅ Tamamlandı!

Repo'nuz artık GitHub'da! 🎉

**Repo URL'iniz:**
```
https://github.com/KULLANICI_ADIN/virtual-tryon-studio
```

## 📝 Sonraki Adımlar (Opsiyonel)

### README'ye Screenshot Ekle

1. Uygulamayı çalıştır: `streamlit run app.py`
2. Ekran görüntüsü al
3. `demo.png` olarak kaydet
4. README.md'nin başına ekle:
```markdown
![Demo](demo.png)
```

### GitHub Actions (CI/CD)

`.github/workflows/test.yml` oluştur:
```yaml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python -m pytest
```

### GitHub Pages (Demo Site)

Settings → Pages → Source: main branch

### Badges Ekle

README.md'ye ekle:
```markdown
![Stars](https://img.shields.io/github/stars/KULLANICI_ADIN/virtual-tryon-studio)
![Issues](https://img.shields.io/github/issues/KULLANICI_ADIN/virtual-tryon-studio)
```

## 🔄 Güncellemeler İçin

```bash
# Değişiklikleri ekle
git add .

# Commit
git commit -m "Güncelleme açıklaması"

# Push
git push
```

## 🆘 Sorun mu var?

### "Permission denied (publickey)"
- HTTPS kullan, SSH yerine
- `git remote set-url origin https://github.com/USER/REPO.git`

### "Updates were rejected"
- `git pull origin main --rebase`
- Sonra `git push`

### Branch problemi
- `git push -u origin main --force` (dikkatli kullan!)

