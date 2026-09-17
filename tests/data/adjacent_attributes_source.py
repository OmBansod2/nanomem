"""Generic, fixture-free personal-memory probe set.

Two shapes:
  ADJACENT  -- two DIFFERENT attributes phrased alike, stored in order; the
               question names one of them. Correct answer = that attribute's
               record. Plain cosine gets these right; a revision layer that
               cannot tell "restatement" from "neighbouring attribute" does not.
  REVISION  -- the SAME attribute stated twice. Correct answer for a "current"
               question = the LATER record, for a "historical" question = the
               earlier one. Plain cosine has no idea; the revision layer is the
               only thing that can get these right.

Nothing here is copied from any benchmark fixture. Round 4 additionally ran a
mechanical token-intersection audit against clean_chat_benchmark.json,
clean_chat_benchmark_heldout.json, golden/chat_v2_golden.json and
benchmark_200_friendship_chat.py and replaced the two distinctive tokens that
collided with the 3-persona selection set; the audit now reports 0.

Run this file to regenerate adjacent_attributes.npz (needs Ollama with
nomic-embed-text). The .npz is what the tests load, so they never touch the
network.
"""

ADJACENT = [
    # (first_text, second_text, question, which_is_answer: 0 or 1)
    ("My dentist is Dr. Aurelio Vance.", "My doctor is Dr. Priya Raghunathan.", "Who is my dentist?", 0),
    ("My dentist is Dr. Aurelio Vance.", "My doctor is Dr. Priya Raghunathan.", "Who is my doctor?", 1),
    ("My home address is 12 Larkspur Way.", "My work address is 44 Steelmill Lane.", "What is my home address?", 0),
    ("My home address is 12 Larkspur Way.", "My work address is 44 Steelmill Lane.", "What is my work address?", 1),
    ("My primary email is quill.harrow@mailbox.test.", "My backup email is spare.harrow@mailbox.test.", "What is my primary email?", 0),
    ("My primary email is quill.harrow@mailbox.test.", "My backup email is spare.harrow@mailbox.test.", "What is my backup email?", 1),
    ("My first language is Marathi.", "My second language is German.", "What is my first language?", 0),
    ("My first language is Marathi.", "My second language is German.", "What is my second language?", 1),
    ("My weekday lunch is a rice bowl.", "My weekend lunch is flatbread and stew.", "What is my weekday lunch?", 0),
    ("My weekday lunch is a rice bowl.", "My weekend lunch is flatbread and stew.", "What is my weekend lunch?", 1),
    ("My savings account is at Corvid Union Bank.", "My current account is at Thornfield Bank.", "Where is my savings account?", 0),
    ("My savings account is at Corvid Union Bank.", "My current account is at Thornfield Bank.", "Where is my current account?", 1),
    ("My mother's name is Ilse Marchetti.", "My father's name is Osric Marchetti.", "What is my mother's name?", 0),
    ("My mother's name is Ilse Marchetti.", "My father's name is Osric Marchetti.", "What is my father's name?", 1),
    ("My morning alarm is set for 6:15.", "My evening alarm is set for 22:40.", "When is my morning alarm?", 0),
    ("My morning alarm is set for 6:15.", "My evening alarm is set for 22:40.", "When is my evening alarm?", 1),
    ("My landlord is Mr. Fenwick Ozu.", "My neighbour is Mrs. Talia Brightwater.", "Who is my landlord?", 0),
    ("My landlord is Mr. Fenwick Ozu.", "My neighbour is Mrs. Talia Brightwater.", "Who is my neighbour?", 1),
    ("My work laptop is a Grellan X14.", "My personal laptop is a Marrow Slate 9.", "What is my work laptop?", 0),
    ("My work laptop is a Grellan X14.", "My personal laptop is a Marrow Slate 9.", "What is my personal laptop?", 1),
    ("My gym membership expires in April.", "My library membership expires in November.", "When does my gym membership expire?", 0),
    ("My gym membership expires in April.", "My library membership expires in November.", "When does my library membership expire?", 1),
    ("My daughter's school is Penrose Grammar.", "My son's school is Aldbury Fields.", "What school does my daughter go to?", 0),
    ("My daughter's school is Penrose Grammar.", "My son's school is Aldbury Fields.", "What school does my son go to?", 1),
    ("My cycling route goes along the Vellum Canal.", "My running route goes through Harrowgate Park.", "Where is my cycling route?", 0),
    ("My cycling route goes along the Vellum Canal.", "My running route goes through Harrowgate Park.", "Where is my running route?", 1),
    ("My travel insurance policy number is TR-88104.", "My home insurance policy number is HM-22317.", "What is my travel insurance policy number?", 0),
    ("My travel insurance policy number is TR-88104.", "My home insurance policy number is HM-22317.", "What is my home insurance policy number?", 1),
    ("My blood group is B negative.", "My partner's blood group is O positive.", "What is my blood group?", 0),
    ("My office desk is on the fourth floor.", "My office locker is on the second floor.", "Where is my office desk?", 0),
    ("My office desk is on the fourth floor.", "My office locker is on the second floor.", "Where is my office locker?", 1),
    ("My wifi password is thistle-moor-41.", "My router admin password is gantry-loom-07.", "What is my wifi password?", 0),
    ("My wifi password is thistle-moor-41.", "My router admin password is gantry-loom-07.", "What is my router admin password?", 1),
    ("My accountant is Ms. Devorah Quint.", "My solicitor is Mr. Ansel Whitlow.", "Who is my accountant?", 0),
    ("My accountant is Ms. Devorah Quint.", "My solicitor is Mr. Ansel Whitlow.", "Who is my solicitor?", 1),
    ("My winter coat is the charcoal parka.", "My rain coat is the olive shell.", "Which is my winter coat?", 0),
    ("My winter coat is the charcoal parka.", "My rain coat is the olive shell.", "Which is my rain coat?", 1),
    ("My bicycle lock combination is 4417.", "My suitcase lock combination is 9052.", "What is my bicycle lock combination?", 0),
    ("My bicycle lock combination is 4417.", "My suitcase lock combination is 9052.", "What is my suitcase lock combination?", 1),
    ("My desk plant is a jade tree.", "My balcony plant is a curry leaf shrub.", "What is my desk plant?", 0),
]

REVISION = [
    # (old_text, new_text, question) -- "current" wants the NEW one
    ("My phone number is 0300 555 1177.", "My phone number is now 0300 555 4820.", "What is my phone number?"),
    ("I live at 12 Larkspur Way.", "I moved to 61 Bellamy Terrace last month.", "Where do I live?"),
    ("My email is quill.harrow@mailbox.test.", "My email changed to q.harrow@postbox.test.", "What is my email address?"),
    ("I work at Corvid Analytics.", "I now work at Thornfield Robotics.", "Where do I work?"),
    ("My car is a blue hatchback.", "I sold the hatchback and bought a grey estate car.", "What car do I drive?"),
    ("My favourite tea is smoked oolong.", "These days my favourite tea is toasted buckwheat.", "What is my favourite tea?"),
    ("My gym is Ironvault on Calder Street.", "I switched my gym to Pinewell Fitness.", "Which gym do I go to?"),
    ("My manager is Beatriks Norleif.", "My manager is now Emeka Dunstable.", "Who is my manager?"),
    ("My passport number is XV4417220.", "My new passport number is ZP9083315.", "What is my passport number?"),
    ("I use a standing desk at home.", "I replaced the standing desk with a wide oak desk.", "What desk do I use at home?"),
    ("My bank is Corvid Union.", "I moved my accounts to Thornfield Bank.", "Which bank do I use?"),
    ("My flight leaves at 09:40.", "My flight was rescheduled to 14:05.", "When does my flight leave?"),
    ("My landlord is Mr. Fenwick Ozu.", "The building changed hands; my landlord is Ms. Ingrid Palewood.", "Who is my landlord?"),
    ("I am allergic to walnuts.", "The new tests say I am allergic to walnuts and sesame.", "What am I allergic to?"),
    ("My laptop is a Grellan X14.", "I upgraded my laptop to a Grellan X22.", "What laptop do I have?"),
    ("My office is on the fourth floor.", "We moved; my office is on the seventh floor now.", "Which floor is my office on?"),
]


if __name__ == "__main__":
    import json
    import os
    import urllib.request

    import numpy as np

    HERE = os.path.dirname(os.path.abspath(__file__))
    texts, index = [], {}

    def idx(t):
        if t not in index:
            index[t] = len(texts)
            texts.append(t)
        return index[t]

    adj = [{"a": idx(a), "b": idx(b), "q": idx(q), "ans": int(k)}
           for a, b, q, k in ADJACENT]
    rev = [{"old": idx(o), "new": idx(n), "q": idx(q)} for o, n, q in REVISION]

    V = np.zeros((len(texts), 768), dtype=np.float32)
    for i, t in enumerate(texts):
        body = json.dumps({"model": "nomic-embed-text", "prompt": t}).encode()
        req = urllib.request.Request("http://localhost:11434/api/embeddings",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            V[i] = np.asarray(json.load(r)["embedding"], dtype=np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    np.savez_compressed(os.path.join(HERE, "adjacent_attributes.npz"),
                        V=V, texts=np.array(texts, dtype=object),
                        adj=json.dumps(adj), rev=json.dumps(rev))
    print(f"{len(texts)} sentences, {len(adj)} adjacent, {len(rev)} revision")
