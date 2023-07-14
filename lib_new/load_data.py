from datasets import load_dataset

initial_prompt = 'Generate a high-quality response for the conversation.\nConversation:'

def make_input(history: list[str]) -> str:
    return '\n'.join((initial_prompt, *history, 'Response:'))

def process_utterance(utterance: list[str]):
    history = []
    it = iter(utterance)
    try:
        while True:
            user = next(it)
            system = next(it)
            history.append('User: ' + user)
            input_ = make_input(history)
            yield (input_, system)
            history.append('System: ' + system)
    except StopIteration:
        pass

def process_utterances(utterances: list[list[str]]):
    for utterance in utterances:
        yield from process_utterance(utterance)

'''
>>> dataset = load_dataset()
>>> print(*dataset[1], sep='\n')
Generate a high-quality response for the conversation.
Conversation:
User: i need a place to dine in the center thats expensive
System: I have several options for you; do you prefer African, Asian, or British food?
User: Any sort of food would be fine, as long as it is a bit expensive. Could I get the phone number for your recommendation?
Response:
There is an Afrian place named Bedouin in the centre. How does that sound?
'''

def load_data(split='train') -> list[tuple[str, str]]:
    dataset = load_dataset('multi_woz_v22', split=split)
    def preprocess(example):
        example['utterances'] = example['turns']['utterance']
        return example
    utterances = dataset.map(preprocess)['utterances']
    del dataset
    return list(process_utterances(utterances))
