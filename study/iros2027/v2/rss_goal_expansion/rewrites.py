"""Conservative text-only clarification memory for the predeclared goal vocabulary.

Rules were engineered with the declared unseen vocabulary. No evaluator case ID
or exact test-string lookup is used. This is a strong authored comparator, not a
blind or learned language-generalization benchmark.
"""
from __future__ import annotations
import re

TARGETS = {
    'cube_behind':"Place the Rubik's cube so that the Rubik's cube is behind the bowl.",
    'glasses_pickup':"Pick up the pair of black eyeglasses that's sitting on the table",
    'mustard_left_bin':'Pick up the mustard bottle and place it in the bin on the left side',
}


def normalize(text):
    s=text.casefold().replace("'",'').replace('’','')
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s)).strip()


def intent(text):
    s=normalize(text)
    if re.search(r'\b(not|dont|never|instead|unless|either|without|avoid|except|leave)\b',s):
        return None
    if not re.search(r'\b(put|place|move|position|pick|lift|grab|raise)\b',s):
        return None
    if re.search(r'\b(glasses|eyeglasses|spectacles)\b',s):
        if re.search(r'\b(cube|bowl|mustard|bin)\b',s):
            return None
        return 'glasses_pickup' if re.search(r'\b(pick up|lift|grab|raise)\b',s) else None
    if 'mustard' in s and 'bin' in s:
        if re.search(r'\b(right|bowl|cube|raisin)\b',s):
            return None
        left_bin = re.search(r'\bleft(?: hand)?(?: side)?\s+bin\b|\bbin\s+on\s+(?:the\s+)?left(?:\s+side)?\b',s)
        return 'mustard_left_bin' if left_bin else None
    if 'cube' in s and 'bowl' in s:
        if re.search(r'\b(banana|mustard|glasses|eyeglasses|spectacles|bin|left|right)\b',s):
            return None
        # Do not turn an instruction to move the reference into a cube repair.
        if re.search(r'\b(move|put|place|position)\s+(?:the\s+)?bowl\b',s):
            return None
        meanings=set()
        for subject,relation,reference in re.findall(
                r'\b(cube|bowl)\b(?:(?!\bcube\b|\bbowl\b).){0,50}?\b(behind|in front of|far side of)\b(?:(?!\bcube\b|\bbowl\b).){0,30}?\b(cube|bowl)\b',s):
            if subject==reference:
                return None
            is_behind = (subject=='cube' and relation in ('behind','far side of')) or (subject=='bowl' and relation=='in front of')
            meanings.add('cube_behind' if is_behind else 'cube_front')
        return 'cube_behind' if meanings=={'cube_behind'} else None
    return None


def route(text,mode='semantic'):
    if mode!='semantic':
        raise ValueError('This cohort declares only the semantic memory comparator')
    key=intent(text)
    effective=TARGETS.get(key,text)
    return {'input_prompt':text,'effective_prompt':effective,'route':key,'rewritten':effective!=text,'mode':mode}
