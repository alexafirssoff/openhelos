import re
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Attempt to import pymorphy2 and set an availability flag
import pymorphy3

morph_analyzer = pymorphy3.MorphAnalyzer()
PYMORPHY_AVAILABLE = True


# ==============================================================================
# Part 1: Core Classes for the Sheaf Model
# ==============================================================================

class SheafSectionProb:
    """Represents a section of the sheaf over a single local context."""
    
    def __init__(self, context_name: str, senses_probs: Dict[str, float], weight: float = 1.0):
        if not math.isclose(sum(senses_probs.values()), 1.0, abs_tol=1e-5):
            raise ValueError(
                f"The sum of probabilities in context '{context_name}' is not equal to 1.0 (sum: {sum(senses_probs.values())})")
        self.context_name = context_name
        self.senses_probs = dict(senses_probs)
        self.weight = weight
    
    def __repr__(self):
        parts = [f"'{sense}': {prob:.2f}" for sense, prob in self.senses_probs.items()]
        return f"SheafSectionProb(context='{self.context_name}', weight={self.weight}, senses={{{', '.join(parts)}}})"


class SheafProb:
    """Represents a probabilistic sheaf for disambiguation."""
    
    def __init__(self):
        self.sections: Dict[str, SheafSectionProb] = {}
    
    def add_section(self, context: str, senses_probs: Dict[str, float], weight: float = 1.0):
        if context in self.sections:
            raise ValueError(f"Context '{context}' already exists in this sheaf.")
        self.sections[context] = SheafSectionProb(context, senses_probs, weight)
    
    def disambiguate(self) -> Tuple[Optional[str], float, Dict[str, Dict[str, float]]]:
        """
        Glues probabilities together and determines the most likely sense.
        Returns a tuple: (best_sense, its_probability, contribution_details).
        """
        total_probs = defaultdict(float)
        contributions = defaultdict(lambda: defaultdict(float))
        
        for context_name, section in self.sections.items():
            for sense, prob in section.senses_probs.items():
                weighted_prob = prob * section.weight
                total_probs[sense] += weighted_prob
                contributions[sense][context_name] = weighted_prob
        
        total_sum = sum(total_probs.values())
        if total_sum == 0:
            return None, 0.0, {}
        
        glued_probs = {sense: prob / total_sum for sense, prob in total_probs.items()}
        sorted_probs = sorted(glued_probs.items(), key=lambda item: item[1], reverse=True)
        
        if not sorted_probs:
            return None, 0.0, {}
        
        best_sense, best_prob = sorted_probs[0]
        return best_sense, best_prob, contributions


# ==============================================================================
# Part 2: Classes for the Word Sense Disambiguation (WSD) Knowledge Base
# ==============================================================================

class WSDKnowledgeBase:
    """Stores the 'dataset' for disambiguating a single word."""
    
    def __init__(self, ambiguous_word: str):
        self.ambiguous_word = ambiguous_word
        self.contexts: Dict[str, Dict[str, float]] = {}
        self.context_weights: Dict[str, float] = {}
    
    def add_context(self, context_word: str, senses_probs: Dict[str, float], weight: float):
        self.contexts[context_word] = senses_probs
        self.context_weights[context_word] = weight


class GlobalKnowledgeBase:
    """A global knowledge base storing information about multiple ambiguous words."""
    
    def __init__(self):
        self._storage: Dict[str, WSDKnowledgeBase] = {}
    
    def add_kb(self, kb: WSDKnowledgeBase):
        self._storage[kb.ambiguous_word] = kb
    
    def get_kb_for_word(self, ambiguous_word: str) -> Optional[WSDKnowledgeBase]:
        return self._storage.get(ambiguous_word)


# ==============================================================================
# Part 3: The Main WSD Engine
# ==============================================================================

class SheafWSD:
    """An engine for word sense disambiguation using the sheaf model."""
    
    def __init__(self, global_knowledge_base: GlobalKnowledgeBase):
        self.global_kb = global_knowledge_base
        self.eng_lemmas = {"flew": "fly", "hit": "hit", "broke": "break", "wooden": "wood"}
        self.negations = {"no", "not", "не", "нет"}
    
    def _lemmatise(self, token: str) -> str:
        """Converts a word to its base form (lemma)."""
        if re.search('[а-яА-Я]', token):  # Check for Cyrillic characters
            if PYMORPHY_AVAILABLE:
                return morph_analyzer.parse(token)[0].normal_form
            return token
        else:
            return self.eng_lemmas.get(token, token)
    
    def _parse_sentence(self, sentence: str) -> List[str]:
        tokens = re.findall(r'\b\w+\b', sentence.lower())
        return [self._lemmatise(token) for token in tokens]
    
    def analyse_text(self, text: str, window_size: int = 4) -> str:
        """
        Analyses a full text, automatically finding and resolving all known
        ambiguous words.
        """
        print(f"Commencing full analysis of the text:\n'''\n{text.strip()}\n'''")
        
        known_ambiguous_words = self.global_kb._storage.keys()
        tokens = self._parse_sentence(text)
        
        found_words_to_resolve = set()
        for token in tokens:
            if token in known_ambiguous_words:
                found_words_to_resolve.add(token)
        
        if not found_words_to_resolve:
            return "No known ambiguous words were found in the text."
        
        full_report = []
        for word in sorted(list(found_words_to_resolve)):
            report_for_word = self._resolve_from_tokens(tokens, text, word, window_size)
            full_report.append(report_for_word)
        
        return "\n\n".join(full_report)
    
    def _resolve_from_tokens(self, tokens: List[str], original_sentence: str, ambiguous_word: str,
                             window_size: int) -> str:
        """An internal helper method that operates on pre-tokenised text."""
        kb = self.global_kb.get_kb_for_word(ambiguous_word)
        indices = [i for i, token in enumerate(tokens) if token == ambiguous_word]
        
        if not indices:
            return ""
        
        results = []
        for i, token_index in enumerate(indices):
            start = max(0, token_index - window_size)
            end = min(len(tokens), token_index + window_size + 1)
            local_window_tokens = tokens[start:token_index] + tokens[token_index + 1:end]
            
            sheaf = SheafProb()
            ignored_contexts = []
            
            for j, token in enumerate(local_window_tokens):
                if token in kb.contexts:
                    # Check if the preceding token in the window is a negation
                    prev_token_index_in_window = j - 1
                    if prev_token_index_in_window >= 0 and local_window_tokens[
                        prev_token_index_in_window] in self.negations:
                        ignored_contexts.append(f"'{local_window_tokens[prev_token_index_in_window]} {token}'")
                        continue
                    
                    senses = kb.contexts[token]
                    weight = kb.context_weights[token]
                    if token not in sheaf.sections:
                        sheaf.add_section(token, senses, weight)
            
            if not sheaf.sections:
                results.append(
                    f"  - Occurrence of '{ambiguous_word}' #{i + 1} (index {token_index}): Insufficient local context.")
                continue
            
            best_sense, probability, contributions = sheaf.disambiguate()
            
            details_str = []
            if ignored_contexts:
                details_str.append(f"    - Ignored (due to negation): {', '.join(ignored_contexts)}")
            
            for sense, context_data in contributions.items():
                parts = [f"'{ctx}'(w={sheaf.sections[ctx].weight:.1f}):{p:.2f}" for ctx, p in context_data.items()]
                details_str.append(
                    f"    - Contributions to sense '{sense}': {sum(context_data.values()):.2f} <= {{{', '.join(parts)}}}")
            
            results.append(
                f"  - Occurrence of '{ambiguous_word}' #{i + 1} (index {token_index}):\n" + "\n".join(details_str) +
                f"\n    VERDICT: '{best_sense}' (final confidence: {probability:.2f})"
            )
        
        return f"--- Analysis for the word '{ambiguous_word}' ---\n" + "\n".join(results)


# ==============================================================================
# Part 4: Building and Populating the Global Knowledge Base
# ==============================================================================

def build_knowledge_base() -> GlobalKnowledgeBase:
    """Creates and populates the global knowledge base from structured data."""
    print("Building the global knowledge base...")
    
    # Structure: {ambiguous_word: {sense: {context: (probability, weight)}}}
    datasets = {
        "bat": {
            "animal": {"animal": (0.9, 1.0), "zoo": (0.95, 1.0), "cave": (0.98, 1.5), "fly": (0.85, 0.8),
                       "night": (0.7, 0.6), "wing": (0.9, 1.2)},
            "sports_tool": {"sport": (0.95, 1.0), "stadium": (0.9, 1.0), "baseball": (0.99, 2.0), "player": (0.95, 1.0),
                            "hit": (0.85, 0.9), "wood": (0.7, 0.7)}
        },
        "кран": {  # Russian: "kran"
            "tap": {"вода": (0.95, 1.2), "кухня": (0.98, 1.0), "ванная": (0.99, 1.0), "капать": (0.9, 1.1),
                    "течь": (0.92, 1.1), "труба": (0.8, 0.8)},
            "crane": {"стройка": (0.99, 1.5), "груз": (0.95, 1.2), "поднимать": (0.98, 1.2), "башенный": (1.0, 2.0),
                      "высокий": (0.8, 0.7), "машина": (0.6, 0.6)}
        },
        "замок": {  # Russian: "zamok"
            "lock": {"дверь": (0.95, 1.2), "ключ": (0.98, 1.5), "закрыть": (0.9, 1.0), "взломать": (0.9, 1.1),
                     "сейф": (0.85, 1.0)},
            "castle": {"король": (0.95, 1.2), "средневековый": (1.0, 2.0), "крепость": (0.98, 1.5),
                       "рыцарь": (0.9, 1.1), "стена": (0.8, 0.7), "башня": (0.85, 0.9)}
        },
        "лук": {  # Russian: "luk"
            "weapon": {"стрела": (1.0, 2.0), "стрелять": (0.95, 1.2), "тетива": (0.98, 1.5), "охота": (0.8, 0.9),
                       "робин": (0.9, 1.0)},
            "vegetable": {"салат": (0.9, 1.0), "суп": (0.85, 1.0), "резать": (0.95, 1.2), "огород": (0.8, 0.8),
                          "зеленый": (0.7, 0.7), "чистить": (0.9, 1.1)}
        }
    }
    
    global_kb = GlobalKnowledgeBase()
    lemmatiser = SheafWSD(None)
    
    for amb_word, senses_data in datasets.items():
        lemmatised_amb_word = lemmatiser._lemmatise(amb_word)
        kb = WSDKnowledgeBase(lemmatised_amb_word)
        all_senses = list(senses_data.keys())
        
        all_context_words = set()
        for sense in all_senses:
            all_context_words.update(senses_data[sense].keys())
        
        for context_word in all_context_words:
            senses_probs = {}
            main_weight = 1.0
            
            for sense in all_senses:
                prob, weight = senses_data[sense].get(context_word, (0.0, 1.0))
                senses_probs[sense] = prob
                if prob > 0:
                    main_weight = weight
            
            total_sum = sum(senses_probs.values())
            if 0 < total_sum < 1.0:
                remaining_mass = 1.0 - total_sum
                zero_prob_senses = [s for s, p in senses_probs.items() if p == 0]
                if zero_prob_senses:
                    per_sense_share = remaining_mass / len(zero_prob_senses)
                    for s in zero_prob_senses:
                        senses_probs[s] = per_sense_share
            
            lemmatised_context = lemmatiser._lemmatise(context_word)
            
            final_sum = sum(senses_probs.values())
            if final_sum > 0:
                normalised_probs = {s: p / final_sum for s, p in senses_probs.items()}
                kb.add_context(lemmatised_context, normalised_probs, main_weight)
        
        global_kb.add_kb(kb)
    
    print(f"Knowledge base has been built and populated for the words: {list(global_kb._storage.keys())}.")
    return global_kb


# ==============================================================================
# Part 5: Demonstration of Functionality
# ==============================================================================
if __name__ == "__main__":
    global_kb = build_knowledge_base()
    wsd_engine = SheafWSD(global_knowledge_base=global_kb)
    separator = "\n" + "=" * 80 + "\n"
    
    # --- Test Case 1: Russian language, multiple ambiguities ---
    text1 = """
    Средневековый рыцарь закрыл дверь на тяжелый замок и пошел чистить свой лук.
    Он знал, что для охоты нужен не тот лук, который режут в салат.
    """
    print(separator)
    report1 = wsd_engine.analyse_text(text1)
    print(report1)
    
    # --- Test Case 2: English language, with negation ---
    text2 = """
    A baseball player hit the ball with a wooden bat. A moment later, a small bat
    flew out of a dark cave. There was no stadium nearby.
    """
    print(separator)
    report2 = wsd_engine.analyse_text(text2)
    print(report2)
    
    # --- Test Case 3: Russian language, a single word with different senses ---
    text3 = "На стройке сломался башенный кран, и рабочему пришлось идти в ванную, чтобы починить текущий кран, так как из него капала вода."
    print(separator)
    report3 = wsd_engine.analyse_text(text3)
    print(report3)