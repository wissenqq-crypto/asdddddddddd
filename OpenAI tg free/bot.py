import asyncio
import logging
import requests
import base64
import json
import os
import time
from datetime import datetime, timedelta
from collections import deque
from io import BytesIO
import threading
import random

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from openai import OpenAI

# ==================== НАСТРОЙКИ ====================
OPENAI_API_KEY = "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"
MAIN_BOT_TOKEN = "7998668144:AAGFVFALh_fESRuORTTx_aa4gpFZZMnHJTY"
ADMIN_BOT_TOKEN = "8227383575:AAHZ_1t3clTl2fKaSgH-X81gM9mSWk55abw"
HELP_BOT_TOKEN = "8571133097:AAEZsQna7qAPjv8Kew3dUHMGCAK22RRmLhk"
ADMIN_ID = 8464509596

MAX_MESSAGE_LENGTH = 4000
REQUEST_TIMEOUT = 120
MAX_HISTORY_LENGTH = 50

# Файлы для хранения данных
SETTINGS_FILE = "bot_settings.json"
USERS_FILE = "bot_users.json"
VIP_FILE = "vip_users.json"
BALANCE_FILE = "user_balance.json"
REFERRAL_FILE = "referral_system.json"
BAN_FILE = "user_bans.json"
MESSAGES_FILE = "user_messages.json"
STATS_FILE = "bot_stats.json"

# Настройки по умолчанию
DEFAULT_SETTINGS = {
    "version": "6.0.0",
    "details": "🌟 Lumina AI - Премиум ИИ помощник",
    "is_blocked": False,
    "block_reason": "",
    "vip_thinking_delay": 2,
    "welcome_bonus": 5,
    "referral_bonus": 10,
    "daily_bonus": 3
}

VIP_PRICES = {
    "week": 50,
    "month": 125,
    "half_year": 200,
    "lifetime": 500
}

# Загрузка данных
def load_data(filename, default=None):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default if default is not None else {}
    return default if default is not None else {}

def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Загрузка всех данных
bot_settings = load_data(SETTINGS_FILE, DEFAULT_SETTINGS)
all_users = load_data(USERS_FILE, [])
vip_users = load_data(VIP_FILE, {})
user_balance = load_data(BALANCE_FILE, {})
referral_system = load_data(REFERRAL_FILE, {})
user_bans = load_data(BAN_FILE, {})
user_messages = load_data(MESSAGES_FILE, {})
bot_stats = load_data(STATS_FILE, {"total_messages": 0, "total_users": 0, "daily_messages": 0})

# ==================== ВСЕ МОДЕЛИ OPENROUTER ====================
MODEL_CONFIGS = [
    # GPT модели
    {
        "name": "GPT-4o Mini",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "openai/gpt-4o-mini",
        "priority": 1,
        "context": 128000
    },
    {
        "name": "GPT-4o",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "openai/gpt-4o",
        "priority": 2,
        "context": 128000
    },
    {
        "name": "Claude 3.5 Sonnet",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "anthropic/claude-3.5-sonnet",
        "priority": 1,
        "context": 200000
    },
    {
        "name": "Gemini 2.0 Flash",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "google/gemini-2.0-flash-exp:free",
        "priority": 1,
        "context": 1048576
    },
    {
        "name": "DeepSeek V3",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "deepseek/deepseek-chat-v3",
        "priority": 1,
        "context": 128000
    },
    {
        "name": "Llama 3.3 70B",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY,
        "model": "meta-llama/llama-3.3-70b-instruct",
        "priority": 2,
        "context": 131072
    }
]

# Модели для медиа
MEDIA_MODELS = {
    "voice": {
        "model": "openai/whisper",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY
    },
    "image_analysis": {
        "model": "google/gemini-2.0-flash-exp:free",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY
    },
    "image_generation": {
        "model": "black-forest-labs/flux-schnell",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": OPENAI_API_KEY
    }
}

# Глобальные переменные
user_conversations = {}
active_clients = {}
model_health = {}
user_last_active = {}

class ConversationManager:
    def __init__(self, max_length=MAX_HISTORY_LENGTH):
        self.max_length = max_length

    def get_conversation(self, user_id):
        if user_id not in user_conversations:
            user_conversations[user_id] = deque(maxlen=self.max_length)
            user_conversations[user_id].append({
                "role": "system",
                "content": """Ты Lumina - премиум AI-помощник нового поколения. Ты умная, дружелюбная и всегда готова помочь.

Твой стиль:
- 💫 Современный и энергичный
- 🤝 Поддерживающий и мотивирующий
- 🎯 Точно отвечаешь на вопросы
- ✨ С легкой долей энтузиазма

Отвечай на русском качественно и с душой!"""
            })
        return user_conversations[user_id]

    def add_message(self, user_id, role, content):
        conversation = self.get_conversation(user_id)
        conversation.append({"role": role, "content": content})

        # Обновляем статистику
        user_id_str = str(user_id)
        user_messages[user_id_str] = user_messages.get(user_id_str, 0) + 1
        bot_stats["total_messages"] = bot_stats.get("total_messages", 0) + 1
        bot_stats["daily_messages"] = bot_stats.get("daily_messages", 0) + 1
        save_data(user_messages, MESSAGES_FILE)
        save_data(bot_stats, STATS_FILE)

    def clear_conversation(self, user_id):
        if user_id in user_conversations:
            user_conversations[user_id] = deque(maxlen=self.max_length)
            user_conversations[user_id].append({
                "role": "system",
                "content": "Ты Lumina - премиум AI-помощник. Отвечай на вопросы качественно и с душой!"
            })

    def get_messages(self, user_id):
        """
        Возвращает список сообщений в формате [{'role':'user'/'assistant'/'system','content':'...'}, ...]
        Используется для передачи моделям.
        """
        conv = self.get_conversation(user_id)
        # Преобразуем deque -> list
        return list(conv)

conversation_manager = ConversationManager()

# ==================== СИСТЕМА БАЛАНСА И VIP ====================
def get_user_balance(user_id):
    return user_balance.get(str(user_id), 0)

def update_user_balance(user_id, amount):
    user_id = str(user_id)
    user_balance[user_id] = user_balance.get(user_id, 0) + amount
    save_data(user_balance, BALANCE_FILE)

def set_user_balance(user_id, amount):
    user_id = str(user_id)
    user_balance[user_id] = int(amount)
    save_data(user_balance, BALANCE_FILE)

def is_vip_user(user_id):
    user_id = str(user_id)
    if user_id not in vip_users:
        return False

    vip_data = vip_users[user_id]
    if vip_data["type"] == "lifetime":
        return True

    end_time = datetime.fromisoformat(vip_data["end_time"])
    return datetime.now() <= end_time

def get_vip_status(user_id):
    user_id = str(user_id)
    if user_id not in vip_users:
        return None

    vip_data = vip_users[user_id]
    if vip_data["type"] == "lifetime":
        return "💎 VIP НАВСЕГДА"

    end_time = datetime.fromisoformat(vip_data["end_time"])
    if datetime.now() > end_time:
        del vip_users[user_id]
        save_data(vip_users, VIP_FILE)
        return None

    time_left = end_time - datetime.now()
    days = time_left.days
    hours = time_left.seconds // 3600
    return f"💎 VIP ({days}д {hours}ч)"

def add_vip_user(user_id, vip_type):
    user_id = str(user_id)
    now = datetime.now()

    duration_map = {
        "week": 7,
        "month": 30,
        "half_year": 180,
        "lifetime": 36500
    }

    end_time = now + timedelta(days=duration_map.get(vip_type, 7))

    vip_users[user_id] = {
        "type": vip_type,
        "start_time": now.isoformat(),
        "end_time": end_time.isoformat()
    }
    save_data(vip_users, VIP_FILE)

# ==================== СИСТЕМА БАНОВ ====================
def is_user_banned(user_id, bot_type="main"):
    user_id = str(user_id)
    if user_id not in user_bans:
        return False

    ban_data = user_bans[user_id].get(bot_type, {})
    if not ban_data:
        return False

    end_time = datetime.fromisoformat(ban_data["end_time"])
    if datetime.now() > end_time:
        del user_bans[user_id][bot_type]
        if not user_bans[user_id]:
            del user_bans[user_id]
        save_data(user_bans, BAN_FILE)
        return False

    return True

# ==================== РЕФЕРАЛЬНАЯ СИСТЕМА ====================
def generate_referral_link(user_id):
    # Возвращаем deep link на бота (если в будущем поменяете username, менять тут)
    return f"https://t.me/LuminaAIBot?start=ref{user_id}"

def get_referral_info(user_id):
    user_id = str(user_id)
    return referral_system.get(user_id, {"referrals": [], "referrer": None, "earned": 0})

def add_referral(referrer_id, referral_id):
    referrer_id = str(referrer_id)
    referral_id = str(referral_id)

    if referrer_id not in referral_system:
        referral_system[referrer_id] = {"referrals": [], "referrer": None, "earned": 0}

    # Проверяем, не приглашал ли уже этот пользователь
    for ref in referral_system[referrer_id]["referrals"]:
        if ref["id"] == referral_id:
            return

    # Добавляем реферала
    referral_system[referrer_id]["referrals"].append({
        "id": referral_id,
        "date": datetime.now().isoformat(),
        "active": True
    })

    # Начисляем бонус
    referral_system[referrer_id]["earned"] += bot_settings["referral_bonus"]
    update_user_balance(int(referrer_id), bot_settings["referral_bonus"])

    save_data(referral_system, REFERRAL_FILE)

# ==================== РАБОТА С МОДЕЛЯМИ ====================
def create_openai_client(config):
    try:
        client = OpenAI(base_url=config["base_url"], api_key=config["api_key"], timeout=30.0)
        return client
    except Exception as e:
        print(f"❌ Ошибка создания клиента {config['name']}: {e}")
        return None

async def test_model_health(client, config):
    try:
        start_time = time.time()
        # Вариант тестового запроса - зависит от SDK, может бросить ошибку, ловим её
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": "Привет"}],
            max_tokens=10,
            temperature=0.1
        )
        response_time = time.time() - start_time
        return True, response_time
    except Exception as e:
        return False, float('inf')

async def initialize_models():
    print("🚀 Запуск всех AI моделей...")

    for config in MODEL_CONFIGS:
        client = create_openai_client(config)
        if client:
            is_healthy, response_time = await test_model_health(client, config)
            if is_healthy:
                active_clients[config["name"]] = client
                model_health[config["name"]] = {
                    "healthy": True,
                    "response_time": response_time,
                    "last_check": datetime.now()
                }
                print(f"✅ {config['name']} - {response_time:.2f}с")
            else:
                print(f"❌ {config['name']} - недоступна")

    print(f"🎯 Готово! {len(active_clients)} моделей активны")

async def get_fastest_response(user_id, user_message):
    # Используем историю через ConversationManager.get_messages
    messages = conversation_manager.get_messages(user_id)
    # Добавляем текущее сообщение в сообщения, если оно ещё не добавлено
    messages_for_model = list(messages) + [{"role": "user", "content": user_message}]

    # Сортируем модели по известной скорости
    healthy_models = []
    for model_name, health_info in model_health.items():
        if health_info.get("healthy"):
            healthy_models.append((model_name, health_info.get("response_time", float('inf'))))

    healthy_models.sort(key=lambda x: x[1])

    for model_name, _ in healthy_models[:3]:
        client = active_clients.get(model_name)
        if client:
            try:
                start_time = time.time()
                response = client.chat.completions.create(
                    model=next(config["model"] for config in MODEL_CONFIGS if config["name"] == model_name),
                    messages=messages_for_model,
                    max_tokens=1200,
                    temperature=0.7
                )

                # Пытаемся взять текст из ответа (SDK-зависимо)
                response_text = ""
                try:
                    response_text = response.choices[0].message.content
                except Exception:
                    try:
                        response_text = response.choices[0].text
                    except Exception:
                        response_text = str(response)

                response_time = time.time() - start_time

                model_health[model_name]["response_time"] = response_time
                model_health[model_name]["last_check"] = datetime.now()

                return response_text

            except Exception as e:
                print(f"❌ Ошибка {model_name}: {e}")
                model_health[model_name]["healthy"] = False
                continue

    # Запасной вариант: пробуем любую модель из конфига
    for config in MODEL_CONFIGS:
        try:
            if config["name"] not in active_clients:
                client = create_openai_client(config)
                if client:
                    active_clients[config["name"]] = client

            client = active_clients.get(config["name"])
            if client:
                response = client.chat.completions.create(
                    model=config["model"],
                    messages=messages_for_model,
                    max_tokens=800,
                    temperature=0.7
                )
                try:
                    return response.choices[0].message.content
                except Exception:
                    try:
                        return response.choices[0].text
                    except Exception:
                        return str(response)
        except Exception:
            continue

    return "⚠️ Система временно перегружена. Попробуйте через минуту! 🕐"

# ==================== РАБОТА С МЕДИА ====================
async def transcribe_voice_message(voice_file):
    try:
        file = await voice_file.get_file()
        file_buffer = BytesIO()
        await file.download_to_memory(file_buffer)
        file_buffer.seek(0)

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=MEDIA_MODELS["voice"]["base_url"], timeout=60.0)

        transcription = client.audio.transcriptions.create(
            model=MEDIA_MODELS["voice"]["model"],
            file=("audio.ogg", file_buffer.read(), "audio/ogg"),
            language="ru"
        )
        # SDK-зависимо: попытка вернуть текст
        try:
            return transcription.text
        except:
            try:
                return transcription["text"]
            except:
                return str(transcription)
    except Exception as e:
        print(f"❌ Ошибка распознавания голоса: {e}")
        return None

async def analyze_image(image_file):
    try:
        file = await image_file.get_file()
        file_buffer = BytesIO()
        await file.download_to_memory(file_buffer)
        file_buffer.seek(0)

        base64_image = base64.b64encode(file_buffer.read()).decode('utf-8')

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=MEDIA_MODELS["image_analysis"]["base_url"], timeout=60.0)

        response = client.chat.completions.create(
            model=MEDIA_MODELS["image_analysis"]["model"],
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Детально опиши что изображено на фотографии. Будь максимально точным и подробным."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }],
            max_tokens=800
        )
        try:
            return response.choices[0].message.content
        except:
            try:
                return response.choices[0].text
            except:
                return str(response)
    except Exception as e:
        print(f"❌ Ошибка анализа изображения: {e}")
        return None

async def generate_image(prompt):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=MEDIA_MODELS["image_generation"]["base_url"], timeout=60.0)

        response = client.images.generate(
            model=MEDIA_MODELS["image_generation"]["model"],
            prompt=prompt,
            n=1,
            size="512x512",
            quality="standard"
        )

        # Попытка получить URL или base64 (SDK-зависимо)
        try:
            image_url = response.data[0].url
            image_response = requests.get(image_url, timeout=30)
            if image_response.status_code == 200:
                return BytesIO(image_response.content)
            return None
        except Exception:
            try:
                # Возможно возвращается base64
                b64 = response.data[0].b64_json
                return BytesIO(base64.b64decode(b64))
            except Exception:
                return None
    except Exception as e:
        print(f"❌ Ошибка генерации изображения: {e}")
        return None

# ==================== ОСНОВНОЙ БОТ ====================
async def main_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    # Реферальная система
    if context.args and context.args[0].startswith('ref'):
        referrer_id = context.args[0][3:]
        if referrer_id and referrer_id != str(user_id):
            add_referral(referrer_id, user_id)

    if bot_settings["is_blocked"]:
        await update.message.reply_text(f"🔧 Бот на техническом обслуживании\n\n{bot_settings['block_reason']}")
        return

    if is_user_banned(user_id, "main"):
        await update.message.reply_text("🚫 Доступ ограничен администрацией")
        return

    # Регистрация нового пользователя
    if user_id not in all_users:
        all_users.append(user_id)
        update_user_balance(user_id, bot_settings["welcome_bonus"])
        bot_stats["total_users"] = len(all_users)
        save_data(all_users, USERS_FILE)
        save_data(bot_stats, STATS_FILE)

    conversation_manager.clear_conversation(user_id)
    user_last_active[user_id] = datetime.now()

    balance = get_user_balance(user_id)
    vip_status = get_vip_status(user_id)

    welcome_text = f"""
🌟 *Добро пожаловать в Lumina AI!*

💫 *Твой статус:* {vip_status if vip_status else '✨ Обычный пользователь'}
💰 *Баланс:* {balance} звезд

🚀 *Что умеет Lumina:*
• 💬 Умные ответы на любые вопросы
• 🎤 Распознавание голоса (VIP)
• 🎨 Генерация изображений (VIP)
• 📷 Анализ фотографий (VIP)
• ⚡ Мгновенные ответы

🎯 *Быстрые команды:*
/ask - Задать вопрос
/balance - Мой баланс
/vip - VIP возможности
/invite - Пригласить друзей
/help - Помощь

💌 *Просто напиши сообщение - и я отвечу!*"""

    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def main_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not update.message or not update.message.text:
        return
    user_message = update.message.text

    if not user_message.strip():
        return

    if is_user_banned(user_id, "main"):
        return

    if bot_settings["is_blocked"]:
        await update.message.reply_text("🔧 Бот временно недоступен")
        return

    user_last_active[user_id] = datetime.now()

    # Добавляем пользователя если новый
    if user_id not in all_users:
        all_users.append(user_id)
        save_data(all_users, USERS_FILE)

    # Задержка для обычных пользователей
    if not is_vip_user(user_id) and bot_settings.get("vip_thinking_delay", 0) > 0:
        thinking_msg = await update.message.reply_text("💭 Думаю...")
        await asyncio.sleep(bot_settings["vip_thinking_delay"])

    # Обработка сообщения
    conversation_manager.add_message(user_id, "user", user_message)

    try:
        response = await asyncio.wait_for(get_fastest_response(user_id, user_message), timeout=REQUEST_TIMEOUT)

        conversation_manager.add_message(user_id, "assistant", response)

        # Удаляем "Думаю..." если было
        if 'thinking_msg' in locals():
            try:
                await thinking_msg.delete()
            except:
                pass

        await update.message.reply_text(response)

    except asyncio.TimeoutError:
        if 'thinking_msg' in locals():
            try:
                await thinking_msg.delete()
            except:
                pass
        await update.message.reply_text("⏰ Время ожидания истекло. Попробуй еще раз!")
    except Exception as e:
        if 'thinking_msg' in locals():
            try:
                await thinking_msg.delete()
            except:
                pass
        await update.message.reply_text("❌ Произошла ошибка. Попробуй еще раз!")

# ==================== КОМАНДЫ ОСНОВНОГО БОТА ====================
async def ask_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("💬 *Задай свой вопрос!*\n\nПросто напиши сообщение ниже 👇", parse_mode='Markdown')

async def balance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    balance = get_user_balance(user_id)
    messages_count = user_messages.get(str(user_id), 0)

    text = f"""
💰 *ТВОЙ БАЛАНС*

💫 *Звезд на счету:* {balance}
📊 *Всего сообщений:* {messages_count}

🎁 *Как пополнить:*
Напиши @helpluminabot для пополнения

💎 *VIP статус дает:*
• Больше возможностей
• Приоритетные ответы
• Медиа-функции"""

    await update.message.reply_text(text, parse_mode='Markdown')

async def vip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    balance = get_user_balance(user_id)
    is_vip = is_vip_user(user_id)

    if is_vip:
        vip_status = get_vip_status(user_id)
        text = f"""
💎 *ТВОЙ VIP СТАТУС*

{vip_status}

🚀 *Твои привилегии:*
• 🎨 Генерация изображений (/gen)
• 🎤 Распознавание голоса
• 📷 Анализ фотографий
• ⚡ Приоритетные ответы
• 🧠 Улучшенный ИИ

✨ Ты уже в VIP-клубе!"""
    else:
        text = f"""
💎 *VIP ПОДПИСКА*

💰 *Твой баланс:* {balance} звезд

🎯 *Варианты подписки:*
• 🟢 1 неделя - 50⭐ (/buy_week)
• 🔵 1 месяц - 125⭐ (/buy_month)
• 🟣 6 месяцев - 200⭐ (/buy_half_year)
• 🟠 НАВСЕГДА - 500⭐ (/buy_lifetime)

🚀 *Что получишь с VIP:*
• 🎨 Генерация изображений
• 🎤 Распознавание голоса
• 📷 Анализ фотографий
• ⚡ Ответы без задержки
• 🧠 Улучшенный ИИ

💫 *Стань частью VIP-клуба!*"""

    await update.message.reply_text(text, parse_mode='Markdown')

async def buy_vip_command(update: Update, context: ContextTypes.DEFAULT_TYPE, vip_type=None):
    # Сделал параметр nullable чтобы можно было вызывать через хендлер-лямбду
    if vip_type is None and context.args:
        vip_type = context.args[0]

    user_id = update.effective_user.id
    balance = get_user_balance(user_id)
    price = VIP_PRICES.get(vip_type, 50)

    vip_names = {
        "week": "1 НЕДЕЛЮ 🟢",
        "month": "1 МЕСЯЦ 🔵",
        "half_year": "6 МЕСЯЦЕВ 🟣",
        "lifetime": "НАВСЕГДА 🟠"
    }

    if balance >= price:
        update_user_balance(user_id, -price)
        add_vip_user(user_id, vip_type)

        await update.message.reply_text(f"""
🎉 *VIP АКТИВИРОВАН!*

{vip_names.get(vip_type, vip_type)}

💫 Добро пожаловать в VIP-клуб!
Теперь тебе доступны все премиум функции!

🚀 Используй:
• /gen - генерация изображений
• Отправляй голосовые сообщения
• Отправляй фото для анализа

✨ Наслаждайся полной версией Lumina!""", parse_mode='Markdown')
    else:
        await update.message.reply_text(f"""
❌ *НЕДОСТАТОЧНО СРЕДСТВ*

💰 Нужно: {price} звезд
💫 У тебя: {balance} звезд

🎁 Для пополнения пиши @helpluminabot""", parse_mode='Markdown')

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_manager.clear_conversation(user_id)
    await update.message.reply_text("🔄 *История диалога очищена!*\n\nНачнем новый разговор! 💫", parse_mode='Markdown')

async def invite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    referral_link = generate_referral_link(user_id)
    ref_info = get_referral_info(user_id)
    referrals_count = len(ref_info.get("referrals", []))
    earned = ref_info.get("earned", 0)

    text = f"""
👥 *ПРИГЛАСИ ДРУЗЕЙ*

🔗 *Твоя ссылка:*
`{referral_link}`

📊 *Статистика:*
👤 Приглашено друзей: {referrals_count}
💰 Заработано звезд: {earned}

🎁 *Как работает:*
1️⃣ Делишься ссылкой с друзьями
2️⃣ Друг переходит по ссылке
3️⃣ Ты получаешь {bot_settings['referral_bonus']} звезд!

💫 *Приглашай больше - получай больше!*"""

    await update.message.reply_text(text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = """
🆘 *ПОМОЩЬ И ПОДДЕРЖКА*

🚀 *Основные команды:*
/start - Перезапустить бота
/ask - Задать вопрос
/balance - Мой баланс
/vip - VIP подписка
/clear - Очистить историю
/invite - Пригласить друзей

🎨 *VIP команды:*
/gen - Генерация изображений
(Отправляй голосовые или фото для анализа)

💰 *Пополнение баланса:*
Пиши @helpluminabot для пополнения

❓ *Есть вопросы?*
Пиши @helpluminabot - поможем!

💫 *Lumina AI - твой умный помощник!*"""

    await update.message.reply_text(text, parse_mode='Markdown')

async def gen_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_vip_user(user_id):
        await update.message.reply_text("""
🎨 *ТОЛЬКО ДЛЯ VIP*

Эта функция доступна только VIP пользователям!

💎 Получи VIP для:
• Генерации изображений
• Распознавания голоса
• Анализа фотографий
• Приоритетных ответов

Используй /vip для покупки!""", parse_mode='Markdown')
        return

    if not context.args:
        await update.message.reply_text("""
🎨 *ГЕНЕРАЦИЯ ИЗОБРАЖЕНИЙ*

💫 *Как использовать:*
`/gen закат над морем в стиле аниме`

🎯 *Примеры запросов:*
• `/gen космонавт в космосе, цифровое искусство`
• `/gen милый котенок в корзинке, фотореалистично`
• `/gen фантастический город будущего`

✨ *Будь креативным!*""", parse_mode='Markdown')
        return

    prompt = ' '.join(context.args)
    await update.message.reply_text("🎨 *Генерирую изображение...*\n\nЭто займет несколько секунд ⏳", parse_mode='Markdown')

    try:
        image_buffer = await generate_image(prompt)
        if image_buffer:
            await update.message.reply_photo(
                photo=image_buffer,
                caption=f"🎨 *Сгенерировано по запросу:*\n{prompt}",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("""
❌ *Не удалось сгенерировать*

Возможно:
• Слишком сложный запрос
• Проблемы с генерацией
• Попробуй другой запрос

✨ Попробуй еще раз!""", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text("❌ Ошибка генерации. Попробуй другой запрос!")

# ==================== ОБРАБОТКА МЕДИА ====================
async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_vip_user(user_id):
        await update.message.reply_text("""
🎤 *ТОЛЬКО ДЛЯ VIP*

Распознавание голоса доступно только VIP!

💎 Получи VIP для:
• Распознавания голоса
• Генерации изображений
• Анализа фотографий

Используй /vip для покупки!""", parse_mode='Markdown')
        return

    await update.message.reply_text("🎤 *Слушаю твое сообщение...*", parse_mode='Markdown')

    try:
        transcription = await transcribe_voice_message(update.message.voice)
        if transcription:
            await update.message.reply_text(f"📝 *Распознанный текст:*\n{transcription}", parse_mode='Markdown')

            conversation_manager.add_message(user_id, "user", f"[Голосовое]: {transcription}")

            response = await get_fastest_response(user_id, transcription)

            conversation_manager.add_message(user_id, "assistant", response)
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("""
❌ *Не удалось распознать*

Возможно:
• Слишком шумно
• Слова неразборчивы
• Попробуй записать четче

✨ Попробуй еще раз!""", parse_mode='Markdown')

    except Exception as e:
        print(f"Ошибка handle_voice_message: {e}")
        await update.message.reply_text("❌ Ошибка распознавания. Попробуй еще раз!")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if not is_vip_user(user_id):
        await update.message.reply_text("""
📷 *ТОЛЬКО ДЛЯ VIP*

Анализ фотографий доступен только VIP!

💎 Получи VIP для:
• Анализа фотографий
• Распознавания голоса
• Генерации изображений

Используй /vip для покупки!""", parse_mode='Markdown')
        return

    await update.message.reply_text("📷 *Анализирую изображение...*", parse_mode='Markdown')

    try:
        analysis = await analyze_image(update.message.photo[-1])
        if analysis:
            await update.message.reply_text(f"📸 *Анализ изображения:*\n{analysis}", parse_mode='Markdown')

            caption = update.message.caption or ""
            media_text = f"[Фото]: {analysis}"
            if caption:
                media_text += f"\n[Подпись]: {caption}"

            conversation_manager.add_message(user_id, "user", media_text)

            response = await get_fastest_response(user_id, media_text)

            conversation_manager.add_message(user_id, "assistant", response)
            await update.message.reply_text(response)
        else:
            await update.message.reply_text("""
❌ *Не удалось проанализировать*

Возможно:
• Изображение слишком сложное
• Проблемы с обработкой
• Попробуй другое фото

✨ Попробуй еще раз!""", parse_mode='Markdown')

    except Exception as e:
        print(f"Ошибка handle_photo_message: {e}")
        await update.message.reply_text("❌ Ошибка анализа. Попробуй другое фото!")

# ==================== БОТ УПРАВЛЕНИЯ ====================
async def admin_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("🚫 Доступ запрещен")
        return

    text = """
🛠️ *ПАНЕЛЬ УПРАВЛЕНИЯ LUMINA AI*

Ниже быстрые кнопки для администрирования. Все действия доступны только владельцу (админу).
Выберите нужную команду или введите её вручную.
"""

    # Клавиатура — кнопки вставляют команду в поле ввода (удобство для админа)
    keyboard = [
        [
            InlineKeyboardButton("📊 Статистика", switch_inline_query_current_chat="/stats"),
            InlineKeyboardButton("👥 Пользователи", switch_inline_query_current_chat="/users")
        ],
        [
            InlineKeyboardButton("💎 VIP список", switch_inline_query_current_chat="/vip_list"),
            InlineKeyboardButton("⚙️ Настройки", switch_inline_query_current_chat="/settings")
        ],
        [
            InlineKeyboardButton("🔄 Перезапуск моделей", switch_inline_query_current_chat="/restart_models"),
            InlineKeyboardButton("📢 Рассылка", switch_inline_query_current_chat="/broadcast ")
        ],
        [
            InlineKeyboardButton("🔐 Блокировать бота", switch_inline_query_current_chat="/block "),
            InlineKeyboardButton("🔓 Разблокировать", switch_inline_query_current_chat="/unblock")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(text, parse_mode='Markdown', reply_markup=reply_markup)

async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    active_vip = sum(1 for user_id in vip_users if is_vip_user(user_id))
    total_balance = sum(int(v) for v in user_balance.values()) if user_balance else 0
    active_today = sum(1 for last_active in user_last_active.values()
                      if (datetime.now() - last_active).total_seconds() < 86400)

    stats_text = f"""
📊 *СТАТИСТИКА LUMINA AI*

👥 *ПОЛЬЗОВАТЕЛИ:*
• Всего пользователей: {len(all_users)}
• Активных за 24ч: {active_today}
• VIP пользователей: {active_vip}

💫 *АКТИВНОСТЬ:*
• Всего сообщений: {bot_stats.get('total_messages', 0)}
• Сообщений сегодня: {bot_stats.get('daily_messages', 0)}
• Активных диалогов: {len(user_conversations)}

💰 *ФИНАНСЫ:*
• Общий баланс: {total_balance}⭐
• Средний баланс: {total_balance/len(all_users) if all_users else 0:.1f}⭐

🎯 *СИСТЕМА:*
• Рабочих моделей: {len(active_clients)}
• Статус бота: {'🟢 АКТИВЕН' if not bot_settings['is_blocked'] else '🔴 ЗАБЛОКИРОВАН'}
• Версия: {bot_settings['version']}"""

    await update.message.reply_text(stats_text, parse_mode='Markdown')

async def admin_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    recent_users = all_users[-15:] if len(all_users) > 15 else all_users
    users_text = "👤 *ПОСЛЕДНИЕ ПОЛЬЗОВАТЕЛИ:*\n\n"

    for user_id in recent_users:
        vip_status = "💎 VIP" if is_vip_user(user_id) else "✨ Обычный"
        balance = get_user_balance(user_id)
        messages = user_messages.get(str(user_id), 0)
        users_text += f"🆔 {user_id} | {vip_status} | {balance}⭐ | {messages} сообщ.\n"

    users_text += f"\n📈 Всего пользователей: {len(all_users)}"

    await update.message.reply_text(users_text, parse_mode='Markdown')

async def admin_user_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /user_info <user_id>")
        return

    user_id = context.args[0]
    user_id_str = str(user_id)

    if user_id_str not in [str(uid) for uid in all_users]:
        await update.message.reply_text("❌ Пользователь не найден")
        return

    balance = get_user_balance(user_id)
    vip_status = get_vip_status(user_id) or "❌ Нет"
    messages = user_messages.get(user_id_str, 0)
    is_banned = is_user_banned(user_id, "main")
    last_active = user_last_active.get(int(user_id), "Неизвестно")

    if isinstance(last_active, datetime):
        last_active = last_active.strftime("%d.%m.%Y %H:%M")

    info_text = f"""
👤 *ИНФОРМАЦИЯ О ПОЛЬЗОВАТЕЛЕ*

🆔 *ID:* {user_id}
💫 *Статус:* {vip_status}
💰 *Баланс:* {balance}⭐
📊 *Сообщений:* {messages}
🚫 *Бан:* {'✅ Да' if is_banned else '❌ Нет'}
🕒 *Последняя активность:* {last_active}

💎 *VIP возможности:* {'✅ Доступны' if is_vip_user(user_id) else '❌ Недоступны'}"""

    await update.message.reply_text(info_text, parse_mode='Markdown')

async def admin_ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text("❌ Использование: /ban <user_id> <дни> [причина]")
        return

    try:
        user_id = context.args[0]
        days = int(context.args[1])
        reason = ' '.join(context.args[2:]) if len(context.args) > 2 else "Нарушение правил"

        user_id_str = str(user_id)
        if user_id_str not in user_bans:
            user_bans[user_id_str] = {}

        end_time = datetime.now() + timedelta(days=days)
        user_bans[user_id_str]["main"] = {
            "reason": reason,
            "end_time": end_time.isoformat(),
            "banned_by": ADMIN_ID,
            "banned_at": datetime.now().isoformat()
        }
        save_data(user_bans, BAN_FILE)

        await update.message.reply_text(f"✅ Пользователь {user_id} забанен на {days} дней\nПричина: {reason}")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

async def admin_unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /unban <user_id>")
        return

    user_id = context.args[0]
    user_id_str = str(user_id)

    if user_id_str in user_bans and "main" in user_bans[user_id_str]:
        del user_bans[user_id_str]["main"]
        if not user_bans[user_id_str]:
            del user_bans[user_id_str]
        save_data(user_bans, BAN_FILE)
        await update.message.reply_text(f"✅ Пользователь {user_id} разбанен")
    else:
        await update.message.reply_text(f"ℹ️ Пользователь {user_id} не забанен")

async def admin_set_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text("❌ Использование: /balance_set <user_id> <сумма>")
        return

    try:
        user_id = context.args[0]
        amount = int(context.args[1])

        # Устанавливаем баланс (а не прибавляем)
        set_user_balance(int(user_id), amount)

        await update.message.reply_text(f"✅ Баланс пользователя {user_id} установлен: {amount}⭐")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

async def admin_vip_add(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args or len(context.args) < 2:
        await update.message.reply_text("❌ Использование: /vip_add <user_id> <тип>\n\nТипы: week, month, half_year, lifetime")
        return

    user_id = context.args[0]
    vip_type = context.args[1]

    if vip_type not in VIP_PRICES:
        await update.message.reply_text("❌ Неверный тип VIP. Доступные: week, month, half_year, lifetime")
        return

    add_vip_user(user_id, vip_type)

    vip_names = {
        "week": "1 неделю",
        "month": "1 месяц",
        "half_year": "6 месяцев",
        "lifetime": "НАВСЕГДА"
    }

    await update.message.reply_text(f"✅ Пользователю {user_id} добавлен VIP на {vip_names[vip_type]}")

async def admin_vip_remove(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /vip_remove <user_id>")
        return

    user_id = context.args[0]
    user_id_str = str(user_id)

    if user_id_str in vip_users:
        del vip_users[user_id_str]
        save_data(vip_users, VIP_FILE)
        await update.message.reply_text(f"✅ VIP у пользователя {user_id} удален")
    else:
        await update.message.reply_text(f"ℹ️ Пользователь {user_id} не имеет VIP")

async def admin_vip_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    active_vip = {uid: data for uid, data in vip_users.items() if is_vip_user(uid)}

    if not active_vip:
        await update.message.reply_text("ℹ️ Нет активных VIP пользователей")
        return

    vip_text = "💎 *АКТИВНЫЕ VIP ПОЛЬЗОВАТЕЛИ:*\n\n"

    for user_id, vip_data in list(active_vip.items())[:20]:  # Первые 20
        vip_type = vip_data["type"]
        if vip_type == "lifetime":
            status = "НАВСЕГДА 🟠"
        else:
            end_time = datetime.fromisoformat(vip_data["end_time"])
            days_left = (end_time - datetime.now()).days
            status = f"{days_left} дней"

        vip_text += f"🆔 {user_id} | {vip_type} | {status}\n"

    vip_text += f"\n📊 Всего VIP: {len(active_vip)}"

    await update.message.reply_text(vip_text, parse_mode='Markdown')

async def admin_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    settings_text = f"""
⚙️ *ТЕКУЩИЕ НАСТРОЙКИ*

🆔 *Версия:* {bot_settings['version']}
🚀 *Статус:* {'🟢 АКТИВЕН' if not bot_settings['is_blocked'] else '🔴 ЗАБЛОКИРОВАН'}
⏰ *Задержка ответа:* {bot_settings.get('vip_thinking_delay', 0)}с

🎁 *БОНУСЫ:*
• Приветственный: {bot_settings.get('welcome_bonus', 0)}⭐
• Реферальный: {bot_settings.get('referral_bonus', 0)}⭐
• Ежедневный: {bot_settings.get('daily_bonus', 0)}⭐

💎 *VIP ЦЕНЫ:*
• Неделя: {VIP_PRICES['week']}⭐
• Месяц: {VIP_PRICES['month']}⭐
• 6 месяцев: {VIP_PRICES['half_year']}⭐
• Навсегда: {VIP_PRICES['lifetime']}⭐

🔄 *Используй команды для изменения настроек*"""

    await update.message.reply_text(settings_text, parse_mode='Markdown')

async def admin_block_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    reason = ' '.join(context.args) if context.args else "Технические работы"

    bot_settings["is_blocked"] = True
    bot_settings["block_reason"] = reason
    save_data(bot_settings, SETTINGS_FILE)

    await update.message.reply_text(f"🔴 Бот заблокирован\nПричина: {reason}")

async def admin_unblock_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    bot_settings["is_blocked"] = False
    bot_settings["block_reason"] = ""
    save_data(bot_settings, SETTINGS_FILE)

    await update.message.reply_text("🟢 Бот разблокирован")

async def admin_set_delay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /set_delay <секунды>")
        return

    try:
        delay = int(context.args[0])
        bot_settings["vip_thinking_delay"] = delay
        save_data(bot_settings, SETTINGS_FILE)

        await update.message.reply_text(f"✅ Задержка ответа установлена: {delay} секунд")
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка: {e}")

async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /broadcast <текст рассылки>")
        return

    message = ' '.join(context.args)
    from telegram import Bot
    main_bot = Bot(token=MAIN_BOT_TOKEN)

    sent = 0
    failed = 0

    await update.message.reply_text(f"📢 Начинаю рассылку для {len(all_users)} пользователей...")

    for user_id in all_users:
        try:
            await main_bot.send_message(chat_id=user_id, text=message, parse_mode='Markdown')
            sent += 1
            await asyncio.sleep(0.1)  # Чтобы не превысить лимиты
        except Exception as e:
            failed += 1
            print(f"❌ Ошибка отправки {user_id}: {e}")

    await update.message.reply_text(f"✅ Рассылка завершена\n\n📤 Отправлено: {sent}\n❌ Ошибок: {failed}")

async def admin_restart_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    await update.message.reply_text("🔄 Перезапускаю AI модели...")

    active_clients.clear()
    model_health.clear()

    await initialize_models()

    await update.message.reply_text(f"✅ Модели перезапущены\nАктивных моделей: {len(active_clients)}")

async def admin_model_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    status_text = "🤖 *СТАТУС AI МОДЕЛЕЙ:*\n\n"

    for model_name, health_info in model_health.items():
        status = "🟢" if health_info["healthy"] else "🔴"
        response_time = health_info["response_time"]
        status_text += f"{status} {model_name} - {response_time:.2f}с\n"

    status_text += f"\n📊 Всего моделей: {len(model_health)}"
    status_text += f"\n🎯 Активных: {sum(1 for h in model_health.values() if h['healthy'])}"

    await update.message.reply_text(status_text, parse_mode='Markdown')

async def admin_active_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    now = datetime.now()
    active_users = []

    for user_id, last_active in user_last_active.items():
        if (now - last_active).total_seconds() < 3600:  # Активны в последний час
            active_users.append((user_id, last_active))

    active_users.sort(key=lambda x: x[1], reverse=True)

    if not active_users:
        await update.message.reply_text("ℹ️ Нет активных пользователей за последний час")
        return

    active_text = "👥 *АКТИВНЫЕ ПОЛЬЗОВАТЕЛИ (последний час):*\n\n"

    for user_id, last_active in active_users[:15]:
        minutes_ago = int((now - last_active).total_seconds() / 60)
        vip_status = "💎" if is_vip_user(user_id) else "✨"
        active_text += f"{vip_status} {user_id} - {minutes_ago} мин. назад\n"

    active_text += f"\n📈 Всего активных: {len(active_users)}"

    await update.message.reply_text(active_text, parse_mode='Markdown')

async def admin_set_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    if not context.args:
        await update.message.reply_text("❌ Использование: /version <новая_версия>")
        return

    new_version = context.args[0]
    bot_settings["version"] = new_version
    save_data(bot_settings, SETTINGS_FILE)

    await update.message.reply_text(f"✅ Версия бота изменена на: {new_version}")

# ==================== БОТ ПОМОЩИ ====================
async def help_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if is_user_banned(user_id, "help"):
        await update.message.reply_text("🚫 Ваш доступ к поддержке ограничен")
        return

    await update.message.reply_text("""
🆘 *СЛУЖБА ПОДДЕРЖКИ LUMINA AI*

💫 Чем могу помочь?

• 💰 Пополнение баланса
• 💎 Покупка VIP статуса
• 🐛 Сообщить об ошибке
• 💡 Предложить улучшение
• ❓ Другой вопрос

📝 Опиши свою проблему подробно - и мы поможем!

⏰ Время ответа: до 24 часов""", parse_mode='Markdown')

async def help_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not update.message or not update.message.text:
        return
    user_message = update.message.text

    if is_user_banned(user_id, "help"):
        return

    # Пересылаем сообщение админу
    try:
        from telegram import Bot
        admin_bot = Bot(token=ADMIN_BOT_TOKEN)

        user_info = f"Пользователь: {user_id}"
        if is_vip_user(user_id):
            user_info += " 💎 VIP"

        await admin_bot.send_message(
            chat_id=ADMIN_ID,
            text=f"📩 *НОВОЕ СООБЩЕНИЕ ОТ ПОЛЬЗОВАТЕЛЯ*\n\n{user_info}\n\n💬 *Сообщение:* {user_message}",
            parse_mode='Markdown'
        )

        await update.message.reply_text("✅ Сообщение отправлено администратору! Ответим в ближайшее время 💫")
    except Exception as e:
        print(f"Ошибка help_handle_message: {e}")
        await update.message.reply_text("❌ Ошибка отправки сообщения")

# ==================== ЗАПУСК БОТОВ ====================
def setup_main_bot():
    application = Application.builder().token(MAIN_BOT_TOKEN).build()

    # Команды
    application.add_handler(CommandHandler("start", main_start))
    application.add_handler(CommandHandler("ask", ask_command))
    application.add_handler(CommandHandler("balance", balance_command))
    application.add_handler(CommandHandler("vip", vip_command))
    application.add_handler(CommandHandler("buy_week", lambda u,c: buy_vip_command(u,c,"week")))
    application.add_handler(CommandHandler("buy_month", lambda u,c: buy_vip_command(u,c,"month")))
    application.add_handler(CommandHandler("buy_half_year", lambda u,c: buy_vip_command(u,c,"half_year")))
    application.add_handler(CommandHandler("buy_lifetime", lambda u,c: buy_vip_command(u,c,"lifetime")))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("invite", invite_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("gen", gen_command))

    # Обработчики сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, main_handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    return application

def setup_admin_bot():
    application = Application.builder().token(ADMIN_BOT_TOKEN).build()

    # Команды админа
    application.add_handler(CommandHandler("start", admin_start))
    application.add_handler(CommandHandler("stats", admin_stats))
    application.add_handler(CommandHandler("users", admin_users))
    application.add_handler(CommandHandler("user_info", admin_user_info))
    application.add_handler(CommandHandler("ban", admin_ban_user))
    application.add_handler(CommandHandler("unban", admin_unban_user))
    application.add_handler(CommandHandler("balance_set", admin_set_balance))
    application.add_handler(CommandHandler("vip_add", admin_vip_add))
    application.add_handler(CommandHandler("vip_remove", admin_vip_remove))
    application.add_handler(CommandHandler("vip_list", admin_vip_list))
    application.add_handler(CommandHandler("settings", admin_settings))
    application.add_handler(CommandHandler("block", admin_block_bot))
    application.add_handler(CommandHandler("unblock", admin_unblock_bot))
    application.add_handler(CommandHandler("set_delay", admin_set_delay))
    application.add_handler(CommandHandler("broadcast", admin_broadcast))
    application.add_handler(CommandHandler("restart_models", admin_restart_models))
    application.add_handler(CommandHandler("model_status", admin_model_status))
    application.add_handler(CommandHandler("active_users", admin_active_users))
    application.add_handler(CommandHandler("version", admin_set_version))

    return application

def setup_help_bot():
    application = Application.builder().token(HELP_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", help_start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, help_handle_message))

    return application

async def daily_cleanup():
    """Ежедневная очистка и сброс статистики"""
    while True:
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            # Сброс дневной статистики
            bot_stats["daily_messages"] = 0
            save_data(bot_stats, STATS_FILE)
            print("🔄 Дневная статистика сброшена")

        await asyncio.sleep(60)  # Проверяем каждую минуту

def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    print("🚀 ЗАПУСК LUMINA AI...")

    # Инициализация моделей
    try:
        asyncio.run(initialize_models())
    except Exception as e:
        print(f"⚠️ Ошибка инициализации моделей: {e}")

    # Запуск ботов в отдельных потоках
    def run_main_bot():
        main_app = setup_main_bot()
        print("🌸 ОСНОВНОЙ БОТ ЗАПУЩЕН")
        main_app.run_polling()

    def run_admin_bot():
        admin_app = setup_admin_bot()
        print("🛠️ БОТ УПРАВЛЕНИЯ ЗАПУЩЕН")
        admin_app.run_polling()

    def run_help_bot():
        help_app = setup_help_bot()
        print("🆘 БОТ ПОДДЕРЖКИ ЗАПУЩЕН")
        help_app.run_polling()

    # Запуск в потоках
    threading.Thread(target=run_main_bot, daemon=True).start()
    threading.Thread(target=run_admin_bot, daemon=True).start()
    threading.Thread(target=run_help_bot, daemon=True).start()

    # Запуск ежедневной очистки
    try:
        asyncio.run(daily_cleanup())
    except Exception as e:
        print(f"⚠️ Ошибка в daily_cleanup: {e}")

    print(f"📊 ПОЛЬЗОВАТЕЛЕЙ: {len(all_users)}")
    print(f"⭐ VIP: {sum(1 for uid in vip_users if is_vip_user(uid))}")
    print(f"🎯 МОДЕЛЕЙ: {len(active_clients)}")
    print("💫 LUMINA AI ЗАПУЩЕНА И ГОТОВА К РАБОТЕ!")

    # Бесконечный цикл
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
