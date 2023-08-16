from typing import Callable

def initialise_translator(deepl_apikey: str, openai_apikey: str) -> Callable:
    import deepl

    from .initialise_gpt4_translator import initialise_gpt4_translator

    deepl_translator = deepl.Translator(deepl_apikey)
    yue2en, en2yue = initialise_gpt4_translator(openai_apikey=openai_apikey)

    lang_code2deepl_lang = {
        'en': 'EN-GB',
        'da': 'DA',
    }

    def translate_text(sentence: str, src_lang: str, dst_lang: str) -> str:
        if src_lang == 'yue':
            assert dst_lang == 'en'
            return yue2en(sentence)
        if dst_lang == 'yue':
            assert src_lang == 'en'
            return en2yue(sentence)

        src_lang = lang_code2deepl_lang[src_lang]
        dst_lang = lang_code2deepl_lang[dst_lang]
        return deepl_translator.translate_text(sentence, source_lang=src_lang, target_lang=dst_lang)

    return translate_text
