#!/usr/bin/env python3
"""
Финальный скрипт для загрузки проекта на GitHub
"""

import os
import sys
import subprocess

def run_command(cmd, description=""):
    """Выполняет команду"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        if result.returncode == 0:
            print(f"[OK] {description}")
            return True
        else:
            print(f"[FAILED] {description}")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"[EXCEPTION] {description}: {e}")
        return False

def main():
    print("FINAL GITHUB UPLOAD")
    print("=" * 50)
    print("Repository: https://github.com/RockInMyHead/face_0711.git")
    print("=" * 50)

    steps = [
        "Initialize Git repository",
        "Create .gitignore and README.md",
        "Add all files",
        "Create initial commit",
        "Add remote origin",
        "Push to GitHub"
    ]

    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}/{len(steps)}: {step}")
        print("-" * 40)

        if step == "Initialize Git repository":
            if not os.path.exists('.git'):
                run_command("git init", step)
            else:
                print("[OK] Git repository already exists")

        elif step == "Create .gitignore and README.md":
            # Создаем .gitignore
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp

# ML Models (downloaded automatically)
.insightface/
*.onnx
*.pb
*.h5

# Test images (keep some for demo)
# Uncomment to ignore all images:
# *.jpg
# *.png
# *.jpeg
# *.bmp
# *.tiff

# Keep demo images
!test_photos/
test_photos/*/
!test_photos/*.jpg
!test_photos/*.png

# Large files
*.zip
*.tar.gz
*.7z

# Database files
*.db
*.sqlite
*.sqlite3
"""

            with open('.gitignore', 'w', encoding='utf-8') as f:
                f.write(gitignore_content)

            # Создаем README.md если его нет
            if not os.path.exists('README.md'):
                readme_content = """# Face Clustering Application

Автоматическая кластеризация фотографий по лицам с использованием AI.

## Особенности

- Распознавание лиц с помощью InsightFace (ArcFace)
- Кластеризация с использованием Faiss
- Веб-интерфейс для управления
- Поддержка различных моделей распознавания
- Групповая обработка общих фотографий

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/RockInMyHead/face_0711.git
cd face_0711
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите сервер:
```bash
python main.py
```

4. Откройте браузер: http://localhost:8000

## Использование

1. Выберите папку с фотографиями
2. Добавьте папку в очередь обработки
3. Нажмите "Обработать очередь"
4. Просматривайте прогресс в разделе "Активные задачи"

## Модели распознавания

- **InsightFace (buffalo_l)** - максимальная точность (рекомендуется)
- **InsightFace (buffalo_s)** - быстрая и легкая
- **Face Recognition (dlib)** - запасной вариант

## Структура проекта

- `main.py` - основной сервер FastAPI
- `cluster_simple.py` - кластеризация с InsightFace
- `cluster_face_recognition.py` - кластеризация с face_recognition
- `static/` - веб-интерфейс
- `requirements.txt` - зависимости Python

## Лицензия

MIT License
"""

                with open('README.md', 'w', encoding='utf-8') as f:
                    f.write(readme_content)

            print("[OK] Created .gitignore and README.md")

        elif step == "Add all files":
            run_command("git add .", step)

        elif step == "Create initial commit":
            # Проверяем, есть ли уже коммиты
            result = subprocess.run("git log --oneline -1", shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode == 0:
                print("[OK] Commits already exist")
            else:
                run_command('git commit -m "Initial commit: Face clustering application with AI"', step)

        elif step == "Add remote origin":
            repo_url = "https://github.com/RockInMyHead/face_0711.git"

            # Проверяем remote
            result = subprocess.run("git remote", shell=True, capture_output=True, text=True, cwd=os.getcwd())
            if "origin" not in result.stdout:
                run_command(f"git remote add origin {repo_url}", step)
            else:
                print("[OK] Remote origin already exists")

        elif step == "Push to GitHub":
            # Пробуем main branch
            run_command("git branch -M main", "Rename branch to main")
            if not run_command("git push -u origin main", step):
                print("Failed to push to main, trying master...")
                run_command("git branch -M master", "Rename branch to master")
                run_command("git push -u origin master", "Push to GitHub (master)")

    print("\n" + "=" * 50)
    print("UPLOAD COMPLETE!")
    print("Repository: https://github.com/RockInMyHead/face_0711.git")
    print("\nNext steps:")
    print("1. Visit the repository on GitHub")
    print("2. Check that all files are uploaded")
    print("3. Add a description and topics")
    print("4. Enable GitHub Pages if needed")

    # Финальная проверка
    print("\nFinal check:")
    run_command("python check_github_upload.py", "Check upload status")

if __name__ == "__main__":
    main()
