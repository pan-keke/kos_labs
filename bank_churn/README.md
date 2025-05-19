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

#### 1. Главная страница (/) / Home Page
- Общая статистика по клиентам
- Быстрый доступ к основным функциям
- Графики и метрики эффективности

#### 2. Список клиентов (/customers/) / Customer List
- Таблица всех клиентов банка
- Сортировка и фильтрация
- Быстрый доступ к детальной информации
- Пагинация для удобного просмотра
- Возможность экспорта данных

#### 3. Карточка клиента (/customers/<id>/) / Customer Details
- Полная информация о клиенте
- История взаимодействий
- График изменения вероятности оттока
- Текущие рекомендации
- Кнопка для генерации нового прогноза

#### 4. История предсказаний (/churn/history/) / Prediction History
- Фильтры по категориям:
  * Все предсказания
  * Клиенты с высоким риском
  * Клиенты с низким риском
- Интерактивные карточки с количеством
- Детальная таблица предсказаний
- Информация о дате, клиенте и вероятности
- Пагинация с сохранением фильтров

#### 5. Рекомендации (/recommendations/list/) / Recommendations
- Фильтры по приоритету:
  * Все рекомендации
  * Высокий приоритет
  * Средний приоритет
  * Низкий приоритет
- Статистика по каждой категории
- Таблица рекомендаций с информацией:
  * Дата создания
  * Клиент
  * Название рекомендации
  * Описание
  * Приоритет
  * Статус активности
- Пагинация с сохранением выбранных фильтров

#### 6. Обучение модели (/churn/train/) / Model Training
- Доступно только администраторам
- Настройка параметров:
  * Количество эпох
  * Размер батча
  * Скорость обучения
  * Валидационное разделение
- Визуализация процесса обучения
- Отображение метрик:
  * Точность (Accuracy)
  * AUC-ROC
  * Потери (Loss)
- История обучений с результатами

#### 7. Управление пользователями (/accounts/users/) / User Management
- Создание и редактирование пользователей
- Управление ролями и правами
- Журнал активности
- Статистика использования системы

## Структура проекта / Project Structure

```
bank_churn/
├── churn_predictor/      # Модуль прогнозирования / Prediction module
│   ├── models.py         # Модели для хранения предсказаний
│   ├── views.py          # Представления для работы с предсказаниями
│   └── templates/        # Шаблоны страниц прогнозирования
├── data_processor/       # Обработка данных / Data processing
│   ├── models.py         # Модели данных клиентов
│   ├── views.py          # Представления для работы с клиентами
│   └── templates/        # Шаблоны страниц клиентов
├── recommendation_system/# Система рекомендаций / Recommendation system
│   ├── models.py         # Модели рекомендаций
│   ├── views.py          # Представления для работы с рекомендациями
│   └── templates/        # Шаблоны страниц рекомендаций
├── accounts/            # Управление пользователями / User management
├── static/             # Статические файлы / Static files
└── templates/          # Общие шаблоны / Common templates
```

## Особенности интерфейса / Interface Features

### Общие элементы / Common Elements
- Адаптивный дизайн для всех устройств
- Анимации для улучшения UX
- Единая цветовая схема в стиле "Барби"
- Интерактивные элементы с обратной связью

### Навигация / Navigation
- Понятное главное меню
- Хлебные крошки для навигации
- Быстрые действия на каждой странице
- Поиск по всем разделам

### Фильтрация и сортировка / Filtering and Sorting
- Интерактивные фильтры
- Сохранение состояния фильтров
- Комбинированная фильтрация
- Сброс фильтров в один клик

## Локализация / Localization

Интерфейс системы полностью на русском языке:
- Все страницы и формы
- Сообщения и уведомления
- Подсказки и описания
- Технические термины с пояснениями

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
