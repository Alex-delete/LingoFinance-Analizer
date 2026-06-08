import os
import json
import requests
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")


def clean_html_content(html_code):
    """
    Вспомогательная функция для очистки HTML от мусора и извлечения чистого текста.
    """
    if not html_code:
        return ""
    soup = BeautifulSoup(html_code, 'html.parser')
    # Удаляем скрипты, стили, мета-теги и навигацию
    for element in soup(["script", "style", "meta", "noscript", "header", "footer", "nav", "aside"]):
        element.decompose()
    clean_text = soup.get_text(separator=' ', strip=True)
    # Сжимаем множественные пробелы
    return ' '.join(clean_text.split())


def get_page_content(url):
    """
    Возвращает ОЧИЩЕННЫЙ текст страницы.
    Если requests возвращает 0 символов полезного текста, автоматически переключается на Playwright.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # Шаг 1: Пробуем быстрый и легкий requests
    print(f"[Requests] Пробуем достучаться до {url}...")
    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200 and "Вы заблокированы" not in response.text:
            print("[Success] Requests выполнил запрос. Проверяем контент...")
            clean_text = clean_html_content(response.text)

            if len(clean_text) > 1000:
                print(f"[Clean] Извлечено {len(clean_text)} символов чистого текста через Requests.")
                return clean_text
            else:
                print("[Warning] Requests вернул 0 полезных символов (пустой JS-каркас). Нужна тяжелая артиллерия...")
        else:
            print(f"[Warning] Requests выдал код {response.status_code} или сработала защита. Переключаемся...")
    except Exception as e:
        print(f"[Error] Requests упал с ошибкой: {e}. Переключаемся на Playwright...")

    # Шаг 2: Если requests выдал 0 символов или упал — запускаем Playwright
    print("[Playwright] Запуск браузера...")
    raw_html = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.wait_for_timeout(5000)
            raw_html = page.content()  # получаем честный HTML после работы JS
            print("[Success] Playwright успешно обошел защиту и забрал HTML!")
        except Exception as pl_error:
            print(f"[Critical] Даже Playwright упал: {pl_error}")
            return None
        finally:
            browser.close()

    if raw_html:
        clean_text = clean_html_content(raw_html)
        print(f"[Clean] Извлечено {len(clean_text)} символов чистого текста через Playwright.")
        return clean_text

    return None


def analyze_bank_text(webpage_text):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    if not API_KEY:
        print("Ошибка: Ключ OPENROUTER_API_KEY не найден в файле .env.")
        return None

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = """Ты — эксперт в области financial лингвоанализа. Из всего предоставленного текста выбери и проанализируй ТОЛЬКО ОДИН продукт (если их несколько — выбери самый заметный или явно указанный). Верни СТРОГО JSON с ключами:
    - product_name
    - creativity_score
    - distortion_index
    - sober_utility_score
    - brand_vibe
    - hidden_catch
    - alternative_suggestion

    Правила заполнения:
    - creativity_score, distortion_index, sober_utility_score — только в формате X/10, where X — число от 0 до 10 (например, "7/10").
    - hidden_catch — чётко опиши, в чём заключается скрытый подвох, риск или манипуляция для малого бизнеса (например, скрытые комиссии, автопродление, привязка к дорогому тарифу).
    - alternative_suggestion — приведи название или краткое описание похожего продукта/решения, который объективно полезнее для малого бизнеса, и объясни почему (1–2 предложения).
    - brand_vibe — 2–4 слова, передающие атмосферу/посыл бренда.
    - Не пиши ничего, кроме JSON. Никаких вступлений, пояснений, текста до или после JSON."""

    MAX_CHARS = 12000
    if len(webpage_text) > MAX_CHARS:
        webpage_text = webpage_text[:MAX_CHARS]

    user_content = f"Проанализируй следующий текст с сайта банка:\n\n{webpage_text}"

    models_pool = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "poolside/laguna-m.1:free",
        "openai/gpt-oss-120b:free",
        "poolside/laguna-xs.2:free",
        "z-ai/glm-4.5-air:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-4-31b-it:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
        "moonshotai/kimi-k2.6:free"
    ]

    for model_name in models_pool:
        print(f"🤖 Пробуем отправить запрос в модель: {model_name}...")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.3,
        }
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            if response.status_code != 200:
                print(f"❌ Модель {model_name} вернула ошибку {response.status_code}. Переключаемся...")
                continue

            raw_result = response.json()
            choices = raw_result.get("choices")
            if not choices:
                continue

            llm_string_output = choices[0].get("message", {}).get("content", "")

            cleaned_output = llm_string_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]
            elif cleaned_output.startswith("```"):
                cleaned_output = cleaned_output[3:]

            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]

            cleaned_output = cleaned_output.strip()
            analysis_dictionary = json.loads(cleaned_output)
            print(f"🎉 Успех! Продукт проанализирован моделью {model_name}.")
            return analysis_dictionary

        except json.JSONDecodeError:
            print(f"⚠️ Ошибка парсинга JSON от {model_name}. Текст ответа не был чистым JSON.")
            continue
        except Exception as e:
            print(f"⚠️ Ошибка сети при работе с {model_name}: {e}. Пробуем следующую...")
            continue

    print("🚨 Ни одна модель из пула не ответила на запрос.")
    return None


if __name__ == "__main__":
    print("=== ДОБРО ПОЖАЛОВАТЬ В БАНКОВСКИЙ АУДИТОР ===")
    user_url = input("Вставь ссылку на банковский продукт: ").strip()

    if not user_url:
        user_url = "https://alfabank.ru/sme/funds/credit-line/"  # исправлено

    raw_text = get_page_content(user_url)

    if raw_text:
        analysis_result = analyze_bank_text(raw_text)
        if analysis_result:
            print("\n=========================================")
            print("     АНАЛИЗ БАНКОВСКОГО ПРОДУКТА (ИИ)    ")
            print("=========================================")
            print(f" Название продукта: {analysis_result.get('product_name')}")
            print(f" Вайб бренда:       {analysis_result.get('brand_vibe')}")
            print(f" Индекс креатива:   {analysis_result.get('creativity_score')}")
            print(f" Уровень обмана:    {analysis_result.get('distortion_index')}")
            print(f" Трезвая полезность:{analysis_result.get('sober_utility_score')}")
            print(f"\n Скрытый подвох:\n ⚠️ {analysis_result.get('hidden_catch')}")
            print(f"\n Альтернатива:\n 💡 {analysis_result.get('alternative_suggestion')}")  # исправлено
            print("=========================================")
        else:
            print("Нейросеть не смогла выполнить анализ.")
    else:
        print("Ошибка: Не удалось собрать текст.")
