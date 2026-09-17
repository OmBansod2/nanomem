"""
Comprehensive 200-Message Friendship Chat Benchmark
====================================================
Simulates meeting a new person and becoming friends over 200 realistic chat turns:
- Casual banter, questions, greetings, reactions, and small talk.
- Personal introductions, career, background, family, pets, hobbies, allergies,
  preferences, contact info, and temporal updates (e.g. changing phone number).

Tests:
1. Purity: Dropping 100% of conversational noise/questions/chaff.
2. Recall: Storing 100% of high-value personal facts and state.
3. Temporal Revision: Updating facts without data collisions.
4. Latency & Memory Footprint across 200 streaming turns.
"""

import os
import sys
import time

# Prioritize local package
cur_dir = os.path.dirname(os.path.abspath(__file__))
if cur_dir not in sys.path:
    sys.path.insert(0, cur_dir)

from nanomem import Vault

# Dataset of 200 conversational messages
# Each entry is: (message, should_store, category_name)
chat_dataset = [
    # --- PHASE 1: Meeting & Initial Introductions (Turns 1-40) ---
    ("Hey there! Good to meet you!", False, "Greeting"),
    ("My name is Alex Mercer.", True, "Identity / Name"),
    ("What brings you to the tech meetup today?", False, "Question"),
    ("I work as a senior game engine developer at Epic.", True, "Career Fact"),
    ("Oh wow, that sounds like an incredible job!", False, "Reaction"),
    ("How long have you been living in San Francisco?", False, "Question"),
    ("I live in the Mission District in San Francisco.", True, "Location Fact"),
    ("Haha nice, the food around the Mission is amazing.", False, "Reaction / banter"),
    ("Do you have a favorite burrito spot?", False, "Question"),
    ("My favorite restaurant is La Taqueria on 24th Street.", True, "Preference"),
    ("Totally agree, their carnitas are legendary!", False, "Reaction"),
    ("Have you always lived on the West Coast?", False, "Question"),
    ("I grew up in Chicago before moving out west.", True, "Background"),
    ("Oh cool, Chicago winters are brutal though, right?", False, "Banter + question"),
    ("Yeah, definitely don't miss the sub-zero snow storms!", False, "Casual agreement"),
    ("I study classical piano in my free time.", True, "Hobby / Skill"),
    ("No way, classical piano? That takes serious dedication.", False, "Reaction"),
    ("Which composers do you like to play most?", False, "Question"),
    ("My favorite composer is Frédéric Chopin.", True, "Preference"),
    ("His nocturnes are breathtaking.", False, "Casual remark"),
    ("Are you an early bird or a night owl?", False, "Question"),
    ("I am definitely a night owl, usually awake until 2 AM.", True, "Lifestyle Fact"),
    ("Lol same here, 1 AM is when I get my best coding done.", False, "Reaction / filler"),
    ("Do you drink a lot of coffee while working?", False, "Question"),
    ("I prefer green tea over coffee actually.", True, "Dietary Preference"),
    ("Interesting! Sencha or matcha?", False, "Question"),
    ("I love Japanese roasted hojicha and genmaicha.", True, "Preference"),
    ("Sounds delicious, I should try that sometime.", False, "Casual filler"),
    ("Do you have any brothers or sisters?", False, "Question"),
    ("I have an older sister named Elena who lives in Seattle.", True, "Family Fact"),
    ("That's awesome, Seattle is such a pretty city.", False, "Reaction"),
    ("What does Elena do up in Seattle?", False, "Question"),
    ("She is a pediatric surgeon at Seattle Children's Hospital.", True, "Family Career"),
    ("Wow, that's such inspiring work.", False, "Reaction"),
    ("Are you allergic to any foods?", False, "Question"),
    ("I am severely allergic to shellfish and lobster.", True, "Health / Allergy"),
    ("Good to know, we'll avoid the seafood buffet then!", False, "Banter"),
    ("Haha yes please, let's keep the EpiPen in the bag!", False, "Joke / laughter"),
    ("Do you have any pets at home?", False, "Question"),
    ("I have a two-year-old golden retriever named Barnaby.", True, "Pet Fact"),

    # --- PHASE 2: Deeper Conversation & Interests (Turns 41-80) ---
    ("Barnaby is such a majestic name for a golden retriever!", False, "Reaction"),
    ("Does Barnaby like playing fetch at the beach?", False, "Question"),
    ("Barnaby loves swimming at Fort Funston beach every Saturday.", True, "Pet Detail"),
    ("Fort Funston is dog paradise with all the sand dunes.", False, "Casual remark"),
    ("What kind of games have you shipped recently?", False, "Question"),
    ("I worked on lighting shaders for Unreal Engine 5 projects.", True, "Technical Experience"),
    ("Rendering shaders is pure wizardry to me honestly.", False, "Reaction"),
    ("Did you study computer graphics in college?", False, "Question"),
    ("I graduated from Northwestern University with a degree in EECS.", True, "Education Fact"),
    ("Northwestern is top tier! Go Wildcats!", False, "Reaction"),
    ("Haha thanks, Evanston was freezing but great memories.", False, "Banter / filler"),
    ("Do you play video games yourself or mostly develop them?", False, "Question"),
    ("I play a lot of tactical RPGs and indie roguelikes.", True, "Hobby / Gaming"),
    ("Have you played Hades or Elden Ring?", False, "Question"),
    ("My favorite video game of all time is Disco Elysium.", True, "Preference"),
    ("Disco Elysium has the greatest writing in gaming history!", False, "Reaction"),
    ("What books have you read lately?", False, "Question"),
    ("I read mostly hard science fiction, especially Greg Egan.", True, "Reading Preference"),
    ("Permutation City blew my mind when I read it.", False, "Casual comment"),
    ("Do you do any outdoor sports besides walking Barnaby?", False, "Question"),
    ("I boulder at Dogpatch Bouldering gym three times a week.", True, "Fitness / Sport"),
    ("Nice, what grade do you climb at?", False, "Question"),
    ("I climb around V6 indoors right now.", True, "Fitness Stat"),
    ("V6 is super solid! Finger strength must be crazy.", False, "Reaction"),
    ("Do you listen to podcasts while climbing or working?", False, "Question"),
    ("I listen to Lex Fridman and Huberman Lab podcasts regularly.", True, "Media Habit"),
    ("Huberman has so much good sleep and dopamine advice.", False, "Casual remark"),
    ("Yeah, morning sunlight and cold showers actually work wonders.", False, "Casual agreement"),
    ("What is your current car or commute setup?", False, "Question"),
    ("I drive a gray 2022 Tesla Model 3.", True, "Asset / Vehicle"),
    ("How do you like the electric commute in SF?", False, "Question"),
    ("Autopilot handles the Bay Bridge traffic like a dream.", True, "Vehicle Detail"),
    ("Bridge traffic is a nightmare otherwise, smart move.", False, "Casual remark"),
    ("Do you speak any foreign languages?", False, "Question"),
    ("I speak fluent French and intermediate Japanese.", True, "Language Skill"),
    ("C'est magnifique! Where did you learn French?", False, "Banter + question"),
    ("My mother was born in Lyon, France.", True, "Heritage Fact"),
    ("Lyon is the culinary capital of France, so lucky!", False, "Reaction"),
    ("Are you planning any trips abroad this year?", False, "Question"),
    ("I am traveling to Kyoto for two weeks in November.", True, "Travel Plan"),

    # --- PHASE 3: Contact Exchange & Work Coordination (Turns 81-120) ---
    ("Kyoto in autumn with the red maple leaves will be stunning.", False, "Reaction"),
    ("We should definitely exchange contacts and grab lunch next week!", False, "Social invitation"),
    ("My mobile number is 415-555-0182.", True, "Personal Phone (Rev 1)"),
    ("Got your number saved! Let me text you right now.", False, "Action filler"),
    ("Did you get my text just now?", False, "Question"),
    ("My email address is alex.mercer.dev@gmail.com.", True, "Email Address"),
    ("Awesome, I will send you the invite to the dev group chat.", False, "Casual remark"),
    ("What is your GitHub handle so I can star your projects?", False, "Question"),
    ("My GitHub handle is @alexmercer-graphics.", True, "Social Handle"),
    ("Following you now! Your Vulkan shader repo looks sick.", False, "Reaction"),
    ("Thanks so much, spent months tuning those compute passes!", False, "Banter"),
    ("Are you on Discord as well?", False, "Question"),
    ("My Discord tag is alex_vulkan#4912.", True, "Discord Handle"),
    ("Sent you a friend request on Discord!", False, "Filler"),
    ("Cool, just accepted your invite.", False, "Filler"),
    ("What timezone are you primarily working in?", False, "Question"),
    ("My timezone is America/Los_Angeles PST.", True, "Timezone Fact"),
    ("Great, same timezone makes async collaboration super easy.", False, "Casual remark"),
    ("Do you have a personal blog or portfolio site?", False, "Question"),
    ("My portfolio website is https://alexmercer.graphics.", True, "Website URL"),
    ("Bookmarking that, the WebGL interactive demos are buttery smooth.", False, "Reaction"),
    ("When is your birthday by the way? We celebrate team birthdays.", False, "Question"),
    ("My birthday is October 14th.", True, "Personal Birthday"),
    ("Libra season! We will get cupcakes for sure.", False, "Banter"),
    ("Haha appreciate it, just no seafood toppings!", False, "Joke"),
    ("What kind of music do you code to?", False, "Question"),
    ("I listen to synthwave and ambient lo-fi while programming.", True, "Music Preference"),
    ("Gunship and Tycho are my go-to artists for deep focus.", False, "Casual recommendation"),
    ("Tycho's Dive album is pure focus fuel, listen to it weekly.", True, "Music Preference"),
    ("Couldn't agree more, timeless album.", False, "Casual remark"),
    ("Do you have any dietary restrictions besides the shellfish allergy?", False, "Question"),
    ("I am also 100% vegetarian.", True, "Dietary Fact"),
    ("That's super easy in SF, so many great vegan and veggie places.", False, "Casual remark"),
    ("Have you tried Shizen vegan sushi in the Mission?", False, "Question"),
    ("Shizen is incredible, the Bodhi roll is out of this world.", True, "Food Review"),
    ("Adding that to my to-eat list immediately.", False, "Reaction"),
    ("What development setup do you use at your desk?", False, "Question"),
    ("I use a custom split mechanical keyboard with Ergodox EZ.", True, "Hardware Preference"),
    ("Ergodox is hardcore ergonomics, wrist health is so important.", False, "Casual remark"),
    ("Yeah, RSI prevention after 10 years of programming is critical.", False, "Casual remark"),

    # --- PHASE 4: Sharing Life Stories & Collaboration (Turns 121-160) ---
    ("How did you get started with computer graphics originally?", False, "Question"),
    ("I started making custom maps and mods for Half-Life 2 in high school.", True, "Background Story"),
    ("Source engine Hammer editor! What a legendary tool.", False, "Reaction"),
    ("Did you always want to make games as a kid?", False, "Question"),
    ("I wanted to be an astrophysicist before falling in love with coding.", True, "Personal History"),
    ("Physics to graphics is such a natural pipeline with all the vector math.", False, "Casual remark"),
    ("Totally, linear algebra and quaternions rule the graphics world.", False, "Casual agreement"),
    ("Do you have any emergency contact info on file for the hackathon?", False, "Question"),
    ("My emergency contact is my sister Elena Mercer at 206-555-0199.", True, "Emergency Contact"),
    ("Noted in the registry! Hopefully we never need to use it.", False, "Banter"),
    ("Haha yes, let's keep all bugs strictly in software!", False, "Joke"),
    ("What code editor do you prefer these days?", False, "Question"),
    ("I use Neovim with Lua configs as my primary editor.", True, "Developer Tool"),
    ("A true Neovim wizard! Vim motions are life changing.", False, "Reaction"),
    ("Can never go back to standard arrow keys after hjkl.", False, "Casual agreement"),
    ("What is your favorite book outside of sci-fi?", False, "Question"),
    ("My favorite non-fiction book is Thinking, Fast and Slow by Kahneman.", True, "Book Preference"),
    ("Cognitive biases and heuristics are fascinating concepts.", False, "Casual remark"),
    ("System 1 vs System 2 thinking changed how I write UI interactions.", True, "Insight / Thought"),
    ("That makes complete sense for UX latency design.", False, "Reaction"),
    ("Do you invest in crypto or traditional index funds?", False, "Question"),
    ("I invest primarily in low-cost Vanguard index funds like VTI.", True, "Financial Fact"),
    ("Bogleheads philosophy! Slow and steady compound interest.", False, "Reaction"),
    ("Simple index funds let me sleep soundly without checking tickers.", False, "Casual remark"),
    ("What is your shoe size? We are ordering custom team sneakers.", False, "Question"),
    ("My shoe size is US Men 10.5.", True, "Attribute"),
    ("Got it, 10.5 logged for the swag drop.", False, "Filler"),
    ("Do you drink alcohol on social nights?", False, "Question"),
    ("I don't drink alcohol, strictly sparkling water or ginger beer.", True, "Lifestyle / Health"),
    ("Respect that! Fever-Tree ginger beer hits the spot every time.", False, "Casual remark"),
    ("Fever-Tree ginger beer with fresh lime is my go-to drink.", True, "Drink Preference"),
    ("Top tier combination right there.", False, "Casual remark"),
    ("What is your favorite season of the year?", False, "Question"),
    ("My favorite season is crisp autumn when leaves turn orange.", True, "Seasonal Preference"),
    ("Autumn sweaters and hoodie weather are unmatched.", False, "Reaction"),
    ("Do you like board games for game nights?", False, "Question"),
    ("I love playing Settlers of Catan and Terraforming Mars.", True, "Hobby / Games"),
    ("Terraforming Mars is epic, takes 3 hours but so rewarding.", False, "Reaction"),
    ("We should organize a game night next Thursday if you're free!", False, "Invitation"),
    ("Count me in, I will bring Terraforming Mars and expansions!", False, "Social agreement"),

    # --- PHASE 5: Updates, Temporal Revisions & Goodbyes (Turns 161-200) ---
    ("Awesome, Thursday night board game session is on the calendar!", False, "Casual confirmation"),
    ("Did you hear about the server migration schedule for our project?", False, "Question"),
    ("Our project staging database server IP is 10.0.4.150.", True, "Project Infrastructure"),
    ("Perfect, I will whitelist my IP against 10.0.4.150.", False, "Work acknowledgment"),
    ("By the way, did your phone carrier finish porting your line?", False, "Question"),
    ("Yes! Actually, I changed my mobile number to 415-555-9921.", True, "Temporal Phone Update (Rev 2)"),
    ("Updated my phone contacts with your new 9921 number!", False, "Acknowledgment"),
    ("Also, remember that our staging API key is sk-stag-774921048.", True, "Security Credential"),
    ("Stored the staging API key safely in my 1Password vault.", False, "Action acknowledgment"),
    ("Are you still living in the Mission District?", False, "Question"),
    ("Actually, I moved to Hayes Valley last weekend.", True, "Temporal Location Update (Rev 2)"),
    ("Hayes Valley is fantastic, so many cool boutiques and cafes.", False, "Reaction"),
    ("Did Barnaby adjust well to the new Hayes Valley apartment?", False, "Question"),
    ("Barnaby loves the new dog park near Alamo Square park.", True, "Pet Update"),
    ("Alamo Square with the Painted Ladies view is iconic.", False, "Casual remark"),
    ("What is your current work title after the promotion?", False, "Question"),
    ("My new title is Principal Graphics Architect at Epic.", True, "Career Promotion Update"),
    ("Huge congratulations on the Principal Architect promotion! Well deserved!", False, "Celebration"),
    ("Thanks so much, really excited for the new engine roadmap.", False, "Polite banter"),
    ("Will you be giving a talk at GDC next spring?", False, "Question"),
    ("I am giving a keynote talk on real-time neural volumetric rendering at GDC.", True, "Upcoming Event"),
    ("I am putting that on my must-attend list for GDC!", False, "Reaction"),
    ("Do you have any recommendations for a great cafe in Hayes Valley?", False, "Question"),
    ("My favorite cafe in Hayes Valley is Ritual Coffee on Octavia.", True, "Preference"),
    ("Ritual makes incredible single-origin pour-overs.", False, "Casual remark"),
    ("Yeah, their Ethiopian washed beans have amazing floral notes.", False, "Banter"),
    ("What time do you usually walk Barnaby in the evening?", False, "Question"),
    ("I walk Barnaby every evening at 6:30 PM.", True, "Routine Fact"),
    ("Maybe we can cross paths at Alamo Square for a dog walk!", False, "Social suggestion"),
    ("That would be great, Barnaby loves meeting friendly humans.", False, "Banter"),
    ("Well, I have to jump on an architecture sync call right now.", False, "Sign-off context"),
    ("Thanks a lot for the wonderful conversation today!", False, "Polite gratitude"),
    ("It was truly great getting to know you!", False, "Social pleasantry"),
    ("Let's definitely catch up on Thursday for board games.", False, "Closing coordination"),
    ("Sounds like a solid plan, see you Thursday at 7 PM!", False, "Casual sign-off"),
    ("Take care and have a productive afternoon!", False, "Pleasantry"),
    ("Bye for now, talk to you soon!", False, "Sign-off"),
    ("Goodbye!", False, "One-word sign-off"),
    ("See ya!", False, "Casual closing"),
    ("Have a great day ahead!", False, "Final pleasantry")
]

print("=" * 85)
print(f"  RUNNING 200-MESSAGE CHAT BENCHMARK: MEETING A NEW FRIEND")
print(f"  Dataset size: {len(chat_dataset)} total turns")
print("=" * 85)

vault_path = "benchmark_200_vault.dat"
if os.path.exists(vault_path):
    os.remove(vault_path)

# Metrics
total_messages = len(chat_dataset)
true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0
durations_ms = []

t_start_total = time.perf_counter()

with Vault(vault_path) as vault:
    for idx, (msg, should_store, cat) in enumerate(chat_dataset, 1):
        t0 = time.perf_counter()
        
        # Run streaming chat turn
        turn = vault.chat(user_message=msg, user_id="user_alex")
        
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        durations_ms.append(elapsed_ms)
        
        did_store = turn["stored"]
        
        if should_store and did_store:
            true_positives += 1
            status = "TP (Stored Fact)  "
            sym = "PASS"
        elif not should_store and not did_store:
            true_negatives += 1
            status = "TN (Dropped Chaff)"
            sym = "PASS"
        elif not should_store and did_store:
            false_positives += 1
            status = "FP (LEAKED NOISE!)"
            sym = "FAIL"
        else: # should_store and not did_store
            false_negatives += 1
            status = "FN (LOST FACT!)   "
            sym = "FAIL"
            
        if sym == "FAIL" or idx in [1, 2, 10, 50, 100, 150, 166, 172, 200]:
            print(f"Turn {idx:3d} | [{sym}] {status} | {elapsed_ms:5.1f}ms | Cat: {cat:<24} | \"{msg[:55]}...\"")

    t_total = time.perf_counter() - t_start_total
    
    print("\n" + "=" * 85)
    print("  CONFUSION MATRIX & ACCURACY SUMMARY (200 MESSAGES)")
    print("=" * 85)
    
    expected_facts = sum(1 for _, s, _ in chat_dataset if s)
    expected_chaff = sum(1 for _, s, _ in chat_dataset if not s)
    correct_total = true_positives + true_negatives
    accuracy = (correct_total / total_messages) * 100.0
    purity = (true_negatives / expected_chaff) * 100.0
    recall = (true_positives / expected_facts) * 100.0
    
    print(f"  • Total Chat Messages Processed : {total_messages}")
    print(f"  • Expected Noise / Questions     : {expected_chaff}")
    print(f"  • Expected High-Value Facts      : {expected_facts}")
    print(f"  • True Negatives (Chaff Filtered): {true_negatives}/{expected_chaff} ({purity:.1f}% Purity)")
    print(f"  • True Positives (Facts Stored)  : {true_positives}/{expected_facts} ({recall:.1f}% Recall)")
    print(f"  • False Positives (Noise Leaked) : {false_positives}")
    print(f"  • False Negatives (Facts Missed) : {false_negatives}")
    print(f"  • Overall Classification Accuracy: {correct_total}/{total_messages} ({accuracy:.1f}%)")
    print(f"  • Total Processing Time          : {t_total:.2f}s (Avg {sum(durations_ms)/len(durations_ms):.2f} ms/turn)")
    
    print("\n" + "=" * 85)
    print("  VERIFYING RETRIEVAL & TEMPORAL UPDATES (MVCC):")
    print("=" * 85)
    
    verification_queries = [
        ("What is Alex's job title and where does he work?", "Principal Graphics Architect at Epic"),
        ("What is Alex's current mobile phone number?", "415-555-9921"),
        ("What was Alex's original phone number?", "415-555-0182"),
        ("Where does Alex live now?", "Hayes Valley"),
        ("What pet does Alex have and what is his name?", "golden retriever named Barnaby"),
        ("What is Alex allergic to?", "shellfish and lobster"),
        ("What is the staging API key?", "sk-stag-774921048"),
        ("Who is Alex's emergency contact?", "Elena Mercer at 206-555-0199"),
        ("What is Alex's favorite video game?", "Disco Elysium"),
        ("What keyboard does Alex use?", "Ergodox EZ")
    ]
    
    for q, expected_snippet in verification_queries:
        temp_dir = "historical" if "original" in q.lower() or "first" in q.lower() else "current"
        results = vault.search(q, top_k=2, temporal_direction=temp_dir)
        top_hit = results[0]["text"] if results else "NONE"
        score = results[0]["score"] if results else 0.0
        rev = results[0].get("revision", 1) if results else 0
        hit = expected_snippet.lower() in top_hit.lower()
        res_sym = "✅ MATCH" if hit else "❌ MISMATCH"
        print(f"\n[Q]: \"{q}\"")
        print(f"     Found [{res_sym}] (Score: {score:.3f}, Rev: {rev}):")
        print(f"     \"{top_hit}\"")
        
    stats = vault.stats()
    print("\n" + "=" * 85)
    print("  VAULT STORAGE METRICS AFTER 200 TURNS:")
    print(f"  • Total Records in Encrypted Vault: {stats['total_documents']}")
    print(f"  • Active Heap Working Set RAM     : {stats['active_heap_ram_kb']} KB")
    print(f"  • Physical Vault File Size        : {stats['file_size_mb']:.3f} MB")
    print(f"  • Encryption Cipher               : {stats['cipher']}")
    print("=" * 85)

if os.path.exists(vault_path):
    os.remove(vault_path)
