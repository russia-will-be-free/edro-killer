import asyncio
from datetime import datetime
import logging
import sys
import os
import shutil
import random
from enum import Enum, unique
from collections import namedtuple
import string
import pickle
from typing import Optional

# let the first request be sync
import telethon.sync
from telethon.tl.patched import Message
from telethon.tl.custom.messagebutton import MessageButton
from telethon import TelegramClient
from telethon import events
from telethon.events.newmessage import NewMessage

project_url = 'https://github.com/russia-will-be-free/edro-killer/'
help_link = f'\nЕсли у вас возникают трудности, уточните на {project_url}issues'
error_link = f'\nСкопируйте все, что вы видите на экране (Кроме первых сообщений!) и отправьте на {project_url}/issues'

try:
    import requests
except ImportError:
    sys.exit(
        'Упс, ошибочка - не хватает установленного пакета.'
        'Пожалуйста, напишите здесь следующее: python3 -m requests\n' + help_link
    )


logger = logging.getLogger(__name__)

MessageParse = namedtuple('MessageParse', ('button_count', 'text'))
Creds = namedtuple('Creds', ('api_id', 'api_hash'))


WAR_BOT_USERNAME = 'er_stopfake_bot'

CREDS_FILENAME = 'creds.rwbf'
SQL_FILENAME = 'россия_будет_свободной.session'


# -------- PARSER ENUMS -------- #

@unique
class Messages(Enum):
    FIRST = MessageParse(2, ('единой россии', 'борьбе с фейками', 'социальных сетях', 'ответьте', 'отправьте обращение'))
    SECOND = MessageParse(0, ('оставьте', 'данные', 'связи', 'фио', 'почта'))
    THIRD = MessageParse(0, ('Вашу', 'электронн', 'почт'))
    FOURTH = MessageParse(0, ('введите', 'вопрос', 'текст' 'обращения'))
    FIFTH = MessageParse(0, ('загрузите' 'материалы', 'наличии', 'скриншоты' 'фотографии', ' др.)'))
    SIXTH = MessageParse(1, ('нажимая', 'согла', 'обработк',
                             '[персональн', 'spb.er.ru/upages/personal'))
    SEVENTH = MessageParse(0, ('спасибо', 'передадим', 'роскомнадзор', 'ВКонтакте', 'Одноклассник'))


@unique
class Buttons(Enum):
    FIRST = (
        ('Социальные сети', 'message-741470-1', 1),
        ('Средства массовой информации', 'message-741470-2', 2),
    )


# -------- HELPERS -------- #


async def random_sleep() -> None:
    sleep = random.randrange(1, random.choice([40, 60, 80, 100])) / 10
    logger.info(f'Сплю {sleep} секунд.')
    await asyncio.sleep(sleep)


def is_message(message: Message, seq: MessageParse) -> bool:
    return any(
        part in message.text.lower() and message.button_count == seq.button_count
        for part in seq.text
    )


class InitialTask:
    _task: asyncio.Future

    def __init__(self) -> None:
        self._task = None

    def has_task(self) -> bool:
        return self._task is not None

    def cancel_task(self) -> None:
        self._task.cancel()

    def create_start_task(self) -> None:
        self._task = asyncio.create_task(start_chatting_async())
        return self._task


Task = InitialTask()


def get_creds() -> Optional[Creds]:
    if os.path.exists(SQL_FILENAME) and os.path.getsize(SQL_FILENAME) > 0:
        return Creds(123, 321)
    if not os.path.exists(CREDS_FILENAME) or not os.path.getsize(CREDS_FILENAME):
        return None
    with open(CREDS_FILENAME, 'r') as f:
        return Creds(*pickle.load(f))


def save_creds(api_id, api_hash: str) -> None:
    with open(CREDS_FILENAME, 'wb') as f:
        pickle.dump((api_id, api_hash), f)


# -------- WORDS GENERATOR -------- #


class SymbolsGenerator:
    def generate_random_word(self, min: int = 2, max: int = 10) -> str:
        ...

    def generate_punctuation_mark(self) -> str:
        return random.choice([' ', ',', '!', '.', ' : ', ' - '])

    def generate_sentence_mark(self) -> str:
        return random.choice([',', ':', ' -', '"', '#', '&'])


class RussianGenerator(SymbolsGenerator):
    aphabet = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪыЬЭэЮюЯя'

    def generate_random_word(self, min: int = 2, max: int = 12) -> str:
        return ''.join([random.choice(self.aphabet) for _ in range(random.choice([min, max]))])

    def generate_random_name(self, min: int = 2, max: int = 12) -> str:
        return ' '.join([self.generate_random_word(min, max) for _ in range(random.choice([2, 4]))])

    def generate_random_sentence(self, max: int = 20) -> str:
        sentence = []
        for _ in range(1, max):
            word = self.generate_random_word()
            if random.random() > 0.8:
                word += self.generate_sentence_mark()
            sentence.append(word)
        return ' '.join(sentence)

    def generate_random_text(self, max=16) -> str:
        return ''.join([f'{self.generate_random_sentence()}{self.generate_punctuation_mark()}' for _ in range(1, max)])


class EnglishGenerator(SymbolsGenerator):
    aphabet = string.ascii_lowercase + string.ascii_uppercase

    def generate_random_word(self, min: int = 2, max: int = 12) -> str:
        return ''.join([random.choice(self.aphabet) for _ in range(random.choice([min, max]))])


class EmailGenerator(EnglishGenerator):
    aphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
    russian = RussianGenerator()

    def generate_random_email(self) -> str:
        name = random.choice([self.russian.generate_random_word(max=64), self.generate_random_word(max=59)])
        domain = random.choice([
            'gmail.com', 'mail.ru', 'mail.com', 'yandex.ru',
            self.russian.generate_random_word(max=6),
            f'{self.generate_random_word(max=5)}.рф',
        ])
        return f'{name}@{domain}'


class RandomImageDownloader:
    URL = 'https://picsum.photos/{height}/{width}?random={seed}'
    PATH = 'image.png'

    def generate_random_url(self) -> str:
        return self.URL.format(
            height=random.choice(range(200, 400)),
            width=random.choice(range(200, 400)),
            seed=random.choice(range(1, 1_000_000)),
        )

    async def download(self) -> bool:
        '''
        # TODO: Make it easy to switch to async without dependency headache
        '''
        try:
            resp = requests.get(self.generate_random_url(), stream=True)
            if resp.status_code != 200:
                return False
            with open(self.PATH, 'wb') as f:
                resp.raw.decode_content = True
                shutil.copyfileobj(resp.raw, f)
                return True
        except Exception:
            logger.exception(f'[{datetime.utcnow()}] Попытался скачать картинку, но вот ошибочка :/' + error_link)
            raise

    async def open_image(self) -> Optional[bytes]:
        try:
            if not os.path.exists('image.png'):
                if not await self.download():
                    return None
            with open(self.PATH, 'rb') as f:
                return f.read()
        except Exception:
            logger.exception(f'[{datetime.utcnow()}] Попытался открыть случайную картинку, но вот ошибочка :-/' + error_link)
            raise


Russian = RussianGenerator()
English = EnglishGenerator()
Email = EmailGenerator()
ImageDownloader = RandomImageDownloader()


# -------- KILLER -------- #


class PutinKillerBot:
    message: Message
    sent = 0

    async def process_message(self, message: Message) -> None:
        self.message = message
        if Task.has_task():
            Task.cancel_task()

        await random_sleep()

        if is_message(message, Messages.FIRST.value):
            self.sent += 1
            print(f'[{datetime.utcnow()}] Раунд {self.sent}')
            await self.first_message()
        elif is_message(message, Messages.SECOND.value):
            await self.second_message()
        elif is_message(message, Messages.THIRD.value):
            await self.third_message()
        elif is_message(message, Messages.FOURTH.value):
            await self.fourth_message()
        elif is_message(message, Messages.FIFTH.value):
            await self.fifth_message()
        elif is_message(message, Messages.SIXTH.value):
            await self.sixth_message()
        elif is_message(message, Messages.SEVENTH.value):
            await Task.create_start_task()
        else:
            print(self.print_message())
            sys.exit(f'[{datetime.utcnow()}] Упс, кажется бота обновили, не могу продолжить милую беседу :(' + error_link)

    async def first_message(self):
        button_to_push: MessageButton = random.choice(self.message.buttons)[0]
        await button_to_push.click(share_phone=False, share_geo=False)

    async def second_message(self):
        await client.send_message(WAR_BOT_USERNAME, Russian.generate_random_name())

    async def third_message(self):
        await client.send_message(WAR_BOT_USERNAME, Email.generate_random_email())

    async def fourth_message(self):
        await client.send_message(WAR_BOT_USERNAME, Russian.generate_random_text())

    async def fifth_message(self):
        '''
        Choose proof.
        '''
        if random.random() > 0.50:
            await ImageDownloader.download()
            image = await ImageDownloader.open_image()
            await client.send_message(
                WAR_BOT_USERNAME,
                file=image,
                force_document=random.choice([True, False]),
                supports_streaming=True,
                message=None if random.random() > 0.90 else Russian.generate_random_text(max=3),
            )
        else:
            await client.send_message(WAR_BOT_USERNAME, Russian.generate_random_sentence(max=3))

    async def sixth_message(self):
        button_to_push: MessageButton = self.message.buttons[0][0]
        await button_to_push.click(share_phone=False, share_geo=False)
        await random_sleep()

    def print_message(self) -> None:
        print(f'[{datetime.utcnow()}] {self.message}')
        print(self.message.text)
        if self.message.buttons is None:
            print(f'[{datetime.utcnow()}] Нет кнопок')
            return
        for btn in self.message.buttons:
            print(btn[0].text)
            print(btn[0].data)
            print(btn[0].inline_query)
            print(btn[0].url)
            print(btn[0].client)


bot = PutinKillerBot()


print(
    'Доброго ранку, ми з Росії. Спасибо, что присоединились, сейчас супер важно'
    'не гундеть про "лучше б не было войны", а что-то по-настоящему делать.'
    'Сейчас вы запустите программу и она начнет уничтожать бота'
    'которого другие (темные) программисты используют для сбора доносов '
    'на тех немногих, кто протестует против войны в России,'
    'Но мы же оба значем, что ябидами быть плохо, да? 😁'
    'Чтобы запустить эту программу, прочитайте инструкцию на: '
    f'{project_url}/README.md'
    + help_link
)

_api_help_text = 'Пожалуйста, вставьте App {} из https://my.telegram.org/apps:\n'

creds = get_creds()

if creds is None:
    API_ID = input(_api_help_text.format('api_id'))
    if not API_ID.isdigit:
        sys.exit(f'[{datetime.utcnow()}] Вы ввели неверный api_id, должно быть число 😥')
    API_HASH = input(_api_help_text.format('api_hash'))
    save_creds(API_ID, API_HASH)
else:
    API_ID, API_HASH = creds


client = TelegramClient('россия_будет_свободной', int(API_ID), API_HASH)
client.start()


@client.on(events.NewMessage(chats=[WAR_BOT_USERNAME]))
async def my_event_handler(event: NewMessage.Event) -> None:
    '''
    Sends the first message to trigger a bot.
    '''
    message: Message = event.to_dict()['message']
    await bot.process_message(message)


def start_chatting():
    return client.send_message(WAR_BOT_USERNAME, '/start')


async def start_chatting_async() -> None:
    await random_sleep()
    await start_chatting()


if __name__ == '__main__':
    print(
        'Ну что-ж, поехали! Совет: чтобы комфортно заниматься своими делами, '
        'нажмите Mute в диалоге с этим ботом, но изредка поглядывайте и проверяйте, '
        'что работа программы продолжается 🙌',
    )
    start_chatting()
    try:
        client.run_until_disconnected()
    finally:
        print(f'\nЗакругляемся. Успели отправить {bot.sent} сообщений. Может быть, еще парочку?')
