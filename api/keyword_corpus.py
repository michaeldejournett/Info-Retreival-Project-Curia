"""Keyword vocabulary ported from backend/keywords.js for retrieval-based expansion."""
from typing import Dict, List

# Each entry: specific_term -> parent_categories
# Ported from backend/keywords.js GENERALIZATIONS
_RULES: List[Dict] = [
    # Food & drink
    {"terms": ["pizza","burger","taco","sushi","pasta","barbecue","bbq","sandwich","buffet","potluck"], "add": ["food","dining"]},
    {"terms": ["coffee","cafe","espresso","latte"], "add": ["food","beverage","cafe"]},
    {"terms": ["beer","brewery","brew","craft beer","ale","lager"], "add": ["food","beverage","alcohol"]},
    {"terms": ["wine","winery","tasting","vineyard","sommelier"], "add": ["food","beverage","alcohol"]},
    {"terms": ["bake","baking","cook","cooking","chef","culinary","recipe"], "add": ["food","cooking"]},
    {"terms": ["dinner","lunch","breakfast","brunch","meal","feast","banquet"], "add": ["food","dining"]},
    # Science
    {"terms": ["biology","botany","zoology","microbiology","ecology"], "add": ["science","stem","biology"]},
    {"terms": ["chemistry","biochemistry","organic chemistry","chemical"], "add": ["science","stem","chemistry"]},
    {"terms": ["physics","quantum","thermodynamics","mechanics"], "add": ["science","stem","physics"]},
    {"terms": ["astronomy","astrophysics","telescope","planet","space","nasa","cosmos"], "add": ["science","stem","space"]},
    {"terms": ["geology","geoscience","earth science","mineralogy"], "add": ["science","stem","geology"]},
    {"terms": ["neuroscience","psychology","cognition","brain"], "add": ["science","stem","health"]},
    {"terms": ["genetics","dna","genome","gene","molecular"], "add": ["science","stem","biology"]},
    {"terms": ["statistics","probability","calculus","algebra","math","mathematics"], "add": ["science","stem","math"]},
    # Technology
    {"terms": ["python","javascript","typescript","java","rust","golang","c++","swift","kotlin"], "add": ["technology","coding","programming"]},
    {"terms": ["machine learning","deep learning","neural network","nlp"], "add": ["technology","ai","stem"]},
    {"terms": ["artificial intelligence","llm","gpt","chatbot"], "add": ["technology","ai","stem"]},
    {"terms": ["robotics","robot","drone","automation"], "add": ["technology","engineering","stem"]},
    {"terms": ["cybersecurity","hacking","ctf","security","pentest"], "add": ["technology","security"]},
    {"terms": ["data science","data analytics","big data","visualization","tableau","pandas"], "add": ["technology","stem","data"]},
    {"terms": ["hackathon","hack","coding challenge","competition"], "add": ["technology","coding"]},
    {"terms": ["web development","frontend","backend","full stack","react","vue","angular"], "add": ["technology","coding","web"]},
    {"terms": ["cloud","aws","azure","gcp","devops","kubernetes","docker"], "add": ["technology","cloud","engineering"]},
    {"terms": ["app","mobile","ios","android"], "add": ["technology","mobile"]},
    # Engineering
    {"terms": ["electrical","circuit","electronics","semiconductor"], "add": ["engineering","stem"]},
    {"terms": ["mechanical","cad","solidworks","3d printing"], "add": ["engineering","stem"]},
    {"terms": ["civil engineering","structural","construction"], "add": ["engineering","stem"]},
    {"terms": ["chemical engineering","process engineering"], "add": ["engineering","stem","chemistry"]},
    # Sports & fitness
    {"terms": ["basketball","nba","dribble","dunk","hoop"], "add": ["sports","athletics"]},
    {"terms": ["soccer","futbol","penalty","goal kick"], "add": ["sports","athletics"]},
    {"terms": ["football","nfl","touchdown","husker"], "add": ["sports","athletics"]},
    {"terms": ["baseball","softball","pitcher","homerun"], "add": ["sports","athletics"]},
    {"terms": ["volleyball","spike","serve"], "add": ["sports","athletics"]},
    {"terms": ["swimming","swim","lap pool","aquatic"], "add": ["sports","fitness","athletics"]},
    {"terms": ["running","marathon","5k","10k","cross country","track"], "add": ["sports","fitness","athletics"]},
    {"terms": ["cycling","bike","bicycle","triathlon"], "add": ["sports","fitness"]},
    {"terms": ["tennis","racket","court"], "add": ["sports","athletics"]},
    {"terms": ["golf","putt","fairway","tee"], "add": ["sports","athletics"]},
    {"terms": ["wrestling","boxing","martial arts","judo","mma","kickboxing"], "add": ["sports","athletics"]},
    {"terms": ["yoga","pilates","stretch"], "add": ["fitness","wellness","health"]},
    {"terms": ["gym","weightlifting","strength training","crossfit"], "add": ["fitness","health"]},
    {"terms": ["rock climbing","bouldering","climbing"], "add": ["sports","fitness","outdoor"]},
    {"terms": ["hiking","trail","backpacking","camping"], "add": ["sports","fitness","outdoor","nature"]},
    # Arts & entertainment
    {"terms": ["painting","watercolor","acrylic","oil paint","canvas"], "add": ["art","visual arts","creative"]},
    {"terms": ["drawing","illustration","sketch","comic"], "add": ["art","visual arts","creative"]},
    {"terms": ["sculpture","ceramics","pottery","clay"], "add": ["art","visual arts","creative"]},
    {"terms": ["photography","photo","camera","portrait"], "add": ["art","creative"]},
    {"terms": ["film","cinema","movie","screening","documentary"], "add": ["art","entertainment","film"]},
    {"terms": ["theater","theatre","play","musical","broadway","improv","drama"], "add": ["art","performance","entertainment"]},
    {"terms": ["dance","ballet","hip hop dance","salsa","ballroom"], "add": ["art","performance","dance"]},
    {"terms": ["concert","live music","gig","band","show"], "add": ["music","performance","entertainment"]},
    {"terms": ["jazz","bebop","blues"], "add": ["music","performance","art"]},
    {"terms": ["opera","symphony","orchestra","classical music","choir","choral"], "add": ["music","performance","art"]},
    {"terms": ["rap","hip hop","r&b","soul music"], "add": ["music","performance","entertainment"]},
    {"terms": ["comedy","stand up","open mic","improv"], "add": ["entertainment","comedy"]},
    {"terms": ["gaming","esports","video game","tabletop","board game"], "add": ["entertainment","gaming"]},
    {"terms": ["poetry","spoken word","literary","writing"], "add": ["art","creative","literature"]},
    {"terms": ["book club","reading","author"], "add": ["education","literature"]},
    # Academic & career
    {"terms": ["lecture","seminar","colloquium","talk"], "add": ["academic","education","learning"]},
    {"terms": ["workshop","training","tutorial","bootcamp"], "add": ["education","learning","skills"]},
    {"terms": ["conference","symposium","summit"], "add": ["academic","professional","networking"]},
    {"terms": ["research","study","experiment","lab","thesis","dissertation"], "add": ["academic","stem","research"]},
    {"terms": ["internship","intern","co-op"], "add": ["career","professional","job"]},
    {"terms": ["networking event","career fair","job fair","recruiter","employer"], "add": ["career","professional","networking"]},
    {"terms": ["resume","cv","job search","interview"], "add": ["career","professional"]},
    {"terms": ["startup","entrepreneur","pitch","venture","founder"], "add": ["career","business","entrepreneurship"]},
    {"terms": ["business","finance","accounting","economics","marketing"], "add": ["business","professional"]},
    {"terms": ["leadership","management","executive"], "add": ["professional","career"]},
    # Health & wellness
    {"terms": ["meditation","mindfulness","breathing","guided"], "add": ["wellness","mindfulness","health"]},
    {"terms": ["mental health","anxiety","stress","depression","therapy","counseling"], "add": ["wellness","health","mental health"]},
    {"terms": ["nutrition","diet","healthy eating","vegan","vegetarian"], "add": ["health","food","wellness"]},
    {"terms": ["first aid","cpr","medical","nursing","healthcare"], "add": ["health","medical"]},
    # Community & social
    {"terms": ["volunteer","volunteering","community service","giving back"], "add": ["community","service","volunteer"]},
    {"terms": ["charity","nonprofit","donation","fundraiser","fundraising"], "add": ["community","nonprofit"]},
    {"terms": ["sustainability","environment","climate","green","eco"], "add": ["environment","community","sustainability"]},
    {"terms": ["recycling","composting","zero waste"], "add": ["environment","community"]},
    {"terms": ["diversity","equity","inclusion","dei","multicultural"], "add": ["community","social","diversity"]},
    {"terms": ["garden","gardening","planting","nature","park"], "add": ["community","outdoor","nature"]},
    {"terms": ["religion","faith","spiritual","church","mosque","temple","prayer"], "add": ["community","spiritual"]},
    {"terms": ["international","culture","cultural","heritage","global"], "add": ["community","culture","international"]},
    {"terms": ["greek life","fraternity","sorority"], "add": ["community","social","student life"]},
    {"terms": ["student org","club","student government","association"], "add": ["community","student life"]},
]

# Flat sorted list of all unique specific terms (the event vocabulary to embed)
SPECIFIC_TERMS: List[str] = sorted({t for rule in _RULES for t in rule["terms"]})

# Map specific term → set of parent categories
TERM_TO_PARENTS: Dict[str, List[str]] = {}
for rule in _RULES:
    for t in rule["terms"]:
        cats = TERM_TO_PARENTS.setdefault(t, [])
        for c in rule["add"]:
            if c not in cats:
                cats.append(c)

# All unique category labels
CATEGORIES: List[str] = sorted({c for rule in _RULES for c in rule["add"]})
