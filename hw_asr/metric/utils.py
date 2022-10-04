# Don't forget to support cases when target_text == ''
from editdistance import distance

def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text)
    return distance(target_text, predicted_text) / len(target_text)



def calc_wer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return len(predicted_text)
    return distance(target_text.split(' '), predicted_text.split(' ')) / len(target_text.split(' '))