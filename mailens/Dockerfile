# Stage 1: Build — с кэшированием зависимостей
FROM python:3.10-slim AS builder

# Системные зависимости
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Python зависимости ГЛОБАЛЬНО
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime — минимальный и безопасный образ
FROM python:3.10-slim

# Системные зависимости (только runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создаём непривилегированного пользователя с домашней директорией
RUN groupadd -r mailuser && \
    useradd -r -g mailuser -d /home/mailuser mailuser && \
    mkdir -p /home/mailuser/.cache

WORKDIR /app

# Копируем зависимости из builder-стадии
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Копируем код приложения
COPY app/ ./app/

# Создаём тестовые письма (15 шт., RU/EN)
RUN mkdir -p test_emails && \
    echo "Subject: Business Proposal\nFrom: partner@company.com\nTo: sales@avsoft.ru\n\nDear Team,\n\nWe propose a partnership to integrate our AI solutions. Let's schedule a call next week.\n\nBest regards,\nAlex Johnson" > test_emails/01_business_en.eml && \
    echo "Subject: Деловое предложение\nFrom: partner@company.ru\nTo: sales@avsoft.ru\n\nУважаемые коллеги,\n\nПредлагаем сотрудничество по внедрению ИИ-решений. Готовы провести презентацию в ближайшее время.\n\nС уважением,\nАлексей Иванов" > test_emails/02_business_ru.eml && \
    echo "Subject: Customer Complaint\nFrom: angry@user.net\nTo: support@avsoft.ru\n\nYour software crashed and I lost all my data! This is unacceptable. I demand compensation immediately!!!\n\nRegards,\nJohn Smith" > test_emails/03_complaint_en.eml && \
    echo "Subject: Жалоба на программное обеспечение\nFrom: user@example.ru\nTo: support@avsoft.ru\n\nПосле обновления ваша программа перестала работать. Я потерял важные файлы. Требую объяснений и компенсации!\n\nС уважением,\nИван Петров" > test_emails/04_complaint_ru.eml && \
    echo "Subject: Technical Support Request\nFrom: dev@client.com\nTo: tech@avsoft.ru\n\nHello,\n\nWe're experiencing issues with API authentication. Error code: 401. Can you help?\n\nThanks,\nMaria" > test_emails/05_support_en.eml && \
    echo "Subject: Запрос в техподдержку\nFrom: admin@client.ru\nTo: tech@avsoft.ru\n\nЗдравствуйте,\n\nПри работе с API возникает ошибка 401. Просьба помочь с настройкой авторизации.\n\nСпасибо,\nАнна" > test_emails/06_support_ru.eml && \
    echo "Subject: CONGRATULATIONS! You Won $1,000,000!!!\nFrom: lottery@fake-prize.org\nTo: you@avsoft.ru\n\nClick here to claim your prize NOW: http://bit.ly/fake-win\n\nHURRY UP! Offer expires in 24 hours!!!" > test_emails/07_spam_caps.eml && \
    echo "Subject: Срочно! Вы выиграли 1 000 000 рублей!\nFrom: prize@win-fast.ru\nTo: you@avsoft.ru\n\nПерейдите по ссылке, чтобы получить приз: http://clck.ru/fake\n\nАкция заканчивается через 24 часа!!!" > test_emails/08_spam_ru.eml && \
    echo "Subject: Urgent: Your account will be suspended\nFrom: security@avsoft-security.ru (FAKE)\nTo: employee@avsoft.ru\n\nDear user,\n\nYour corporate account will be blocked in 1 hour. Verify now: http://avsoft-login.ru/fake\n\nIT Department" > test_emails/09_phishing_en.eml && \
    echo "Subject: Важно: ваша учётная запись будет заблокирована\nFrom: support@avsoft-support.ru (ПОДДЕЛКА)\nTo: employee@avsoft.ru\n\nУважаемый сотрудник,\n\nВаша корпоративная учётная запись будет заблокирована через 1 час. Подтвердите данные: http://avsoft-login.ru/fake\n\nСлужба поддержки" > test_emails/10_phishing_ru.eml && \
    echo "Subject: Job Offer — Senior Data Engineer\nFrom: hr@techcorp.com\nTo: you@avsoft.ru\n\nWe invite you for an interview for Senior Data Engineer position. Salary: 300 000 RUB.\n\nBest,\nHR Team" > test_emails/11_hr_en.eml && \
    echo "Subject: Вакансия: Старший инженер данных\nFrom: hr@techcorp.ru\nTo: you@avsoft.ru\n\nПриглашаем вас на собеседование на позицию Старшего инженера данных. ЗП: 300 000 руб.\n\nС уважением,\nКоманда HR" > test_emails/12_hr_ru.eml && \
    echo "Subject: Invoice #INV-2025-789\nFrom: accounting@vendor.com\nTo: finance@avsoft.ru\n\nPlease find attached invoice for Q4 services. Amount: $15,000. Payment due in 10 days.\n\nRegards,\nFinance Dept" > test_emails/13_finance_en.eml && \
    echo "Subject: Счёт №СЧ-2025-789\nFrom: billing@vendor.ru\nTo: finance@avsoft.ru\n\nВо вложении счёт за услуги Q4. Сумма: 1 500 000 руб. Оплата в течение 10 дней.\n\nС уважением,\nБухгалтерия" > test_emails/14_finance_ru.eml && \
    echo "Subject: Meeting: Project Sync — Dec 26, 14:00 MSK\nFrom: pm@avsoft.ru\nTo: team@avsoft.ru\n\nAgenda:\n- Sprint review\n- Roadmap Q1\n- Risks\n\nJoin: https://meet.google.com/abc-defg-hij" > test_emails/15_meeting.en.eml

# Few-shot примеры
RUN mkdir -p few_shot_examples/complaint few_shot_examples/spam few_shot_examples/business && \
    echo "I am extremely dissatisfied with your service quality." > few_shot_examples/complaint/en1.txt && \
    echo "Ваш продукт не работает, я требую возврата денег!" > few_shot_examples/complaint/ru1.txt && \
    echo "WIN NOW! CLICK HERE TO GET YOUR PRIZE IMMEDIATELY!!!" > few_shot_examples/spam/en1.txt && \
    echo "СРОЧНО! ВЫ ВЫИГРАЛИ! НАЖМИТЕ СЮДА СЕЙЧАС!!!" > few_shot_examples/spam/ru1.txt && \
    echo "We propose a strategic partnership for AI integration." > few_shot_examples/business/en1.txt && \
    echo "Предлагаем стратегическое партнёрство по внедрению ИИ." > few_shot_examples/business/ru1.txt

# Устанавливаем NLTK данные (под root)
RUN python -m nltk.downloader punkt averaged_perceptron_tagger

# >>> КРИТИЧЕСКИ ВАЖНО: chown ДО переключения пользователя <<<
RUN chown -R mailuser:mailuser /app /home/mailuser

# Переключаемся на mailuser
USER mailuser

# Проверяем, что можем писать в кэш
RUN mkdir -p /home/mailuser/.cache && touch /home/mailuser/.cache/.writable-test

# Переменные окружения
ENV \
    HOME=/home/mailuser \
    TRANSFORMERS_CACHE=/home/mailuser/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/mailuser/.cache/torch/sentence_transformers \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Порты
EXPOSE 8501

# Health-check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health 2>/dev/null || exit 1

# Запуск
CMD ["streamlit", "run", "app/ui.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.maxUploadSize=200"]