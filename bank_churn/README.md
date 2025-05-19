# Система анализа оттока клиентов банка / Bank Customer Churn Analysis System

## О проекте / About

Система прогнозирования оттока клиентов банка с использованием нейронных сетей. Проект включает веб-интерфейс на русском языке для анализа и предсказания вероятности ухода клиентов.

Bank customer churn prediction system using neural networks. The project includes a Russian web interface for analyzing and predicting customer churn probability.

### Основные функции / Key Features

- 🔄 Прогнозирование оттока клиентов в реальном времени / Real-time customer churn prediction
- 📊 Анализ исторических данных / Historical data analysis
- 👥 Управление данными клиентов / Customer data management
- 📈 Обучение модели с визуализацией / Model training with visualization
- 📋 История предсказаний / Prediction history
- 💡 Система рекомендаций / Recommendation system

### Технологии / Technologies

- Python 3.8+
- Django 4.x
- TensorFlow 2.x
- Bootstrap 5
- jQuery
- SQLite/PostgreSQL

## Установка / Installation

```bash
# Клонирование репозитория / Clone the repository
git clone https://github.com/yourusername/bank-churn-prediction.git
cd bank-churn-prediction

# Создание виртуального окружения / Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установка зависимостей / Install dependencies
pip install -r requirements.txt

# Миграции базы данных / Database migrations
python manage.py migrate

# Запуск сервера разработки / Run development server
python manage.py runserver
```

## Использование / Usage

### Роли пользователей / User Roles

- 👨‍💼 Администратор / Administrator
  - Обучение модели / Model training
  - Управление пользователями / User management
  - Доступ к журналу активности / Access to activity log

- 👨‍💻 Менеджер / Manager
  - Ввод данных клиентов / Customer data input
  - Просмотр предсказаний / View predictions
  - Генерация рекомендаций / Generate recommendations

### Основные страницы / Main Pages

- 📝 Ввод данных клиента / Customer Data Input
- 📊 Список клиентов / Customer List
- 📈 История предсказаний / Prediction History
- 🎯 Обучение модели / Model Training
- 👥 Управление пользователями / User Management

### Описание страниц / Pages Description

#### Ввод данных клиента / Customer Data Input
- Форма ввода информации о клиенте
- Поля для ввода: кредитный рейтинг, регион, пол, возраст, срок обслуживания
- Дополнительные параметры: баланс, количество продуктов, наличие кредитной карты
- Автоматическая валидация данных
- Мгновенный расчет вероятности оттока

#### Список клиентов / Customer List
- Таблица всех клиентов банка
- Сортировка по различным параметрам
- Фильтрация по статусу активности
- Быстрый доступ к детальной информации
- Пагинация для удобного просмотра

#### История предсказаний / Prediction History
- Журнал всех сделанных предсказаний
- Статистика по высокому/низкому риску
- Временная шкала предсказаний
- Возможность фильтрации по дате и результату
- Детальная информация о каждом предсказании

#### Обучение модели / Model Training
- Интерфейс для администраторов
- Настройка параметров обучения:
  * Количество эпох (Epochs)
  * Размер батча (Batch Size)
  * Скорость обучения (Learning Rate)
  * Валидационное разделение (Validation Split)
- Визуализация процесса обучения
- Метрики производительности модели (AUC, Accuracy)

#### Детали клиента / Customer Details
- Полная информация о клиенте
- История взаимодействий
- График изменения вероятности оттока
- Система рекомендаций для удержания
- Возможность обновления данных

#### Управление пользователями / User Management
- Создание новых пользователей
- Управление ролями и правами
- Журнал активности пользователей
- Сброс паролей и блокировка аккаунтов
- Статистика использования системы

## Структура проекта / Project Structure

```
bank_churn/
├── churn_predictor/      # Модуль прогнозирования / Prediction module
├── data_processor/       # Обработка данных / Data processing
├── recommendation_system/# Система рекомендаций / Recommendation system
├── accounts/            # Управление пользователями / User management
├── static/             # Статические файлы / Static files
└── templates/          # Шаблоны / Templates
```

## Локализация / Localization

Интерфейс системы полностью переведен на русский язык. Основные элементы:
- Все шаблоны и формы
- Сообщения об ошибках и уведомления
- Технические термины сопровождаются пояснениями

The system interface is fully translated to Russian. Main elements:
- All templates and forms
- Error messages and notifications
- Technical terms are accompanied by explanations

## Лицензия / License

MIT

## Поддержка / Support

По вопросам работы системы обращайтесь: your.email@example.com 

# URLs проекта / Project URLs

## Основные URL-пути / Main URLs

### Управление клиентами / Customer Management
- `/customers/` - Список всех клиентов / List of all customers
- `/customers/add/` - Добавление нового клиента / Add new customer
- `/customers/<id>/` - Детальная информация о клиенте / Customer details
- `/customers/<id>/edit/` - Редактирование данных клиента / Edit customer data
- `/customers/input/` - Форма ввода данных клиента / Customer data input form

### Прогнозирование / Prediction
- `/churn/predict/<id>/` - Расчет вероятности оттока для клиента / Calculate churn probability
- `/churn/history/` - История всех предсказаний / Prediction history
- `/churn/train/` - Страница обучения модели (только для админов) / Model training page (admin only)

### Рекомендации / Recommendations
- `/recommendations/generate/<id>/` - Генерация рекомендаций для клиента / Generate recommendations
- `/recommendations/history/<id>/` - История рекомендаций для клиента / Recommendation history

### Управление пользователями / User Management
- `/accounts/login/` - Страница входа / Login page
- `/accounts/logout/` - Выход из системы / Logout
- `/accounts/register/` - Регистрация нового пользователя / Register new user
- `/accounts/profile/` - Профиль пользователя / User profile
- `/accounts/users/` - Управление пользователями (админ) / User management (admin)
- `/accounts/activity/` - Журнал активности / Activity log

### API Endpoints
- `/api/customers/` - API для работы с клиентами / Customer API
- `/api/predictions/` - API для получения предсказаний / Predictions API
- `/api/recommendations/` - API для работы с рекомендациями / Recommendations API

### Дополнительные страницы / Additional Pages
- `/` - Главная страница / Home page
- `/dashboard/` - Панель управления / Dashboard
- `/stats/` - Статистика и аналитика / Statistics and analytics
- `/help/` - Справка и документация / Help and documentation

## Права доступа / Access Rights

### Администратор / Administrator
- Доступ ко всем URL / Access to all URLs
- Обучение модели / Model training
- Управление пользователями / User management
- Просмотр журнала активности / View activity log

### Менеджер / Manager
- Работа с клиентами / Customer management
- Просмотр предсказаний / View predictions
- Генерация рекомендаций / Generate recommendations
- Базовая статистика / Basic statistics

### Оператор / Operator
- Просмотр клиентов / View customers
- Базовые предсказания / Basic predictions
- Просмотр рекомендаций / View recommendations 