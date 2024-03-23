reasoning_prompt = {
   "system": "Given a question and the associated retrieved reasoning path from a knowledge graph, you are asked to answer the following question based on the reasoning path and your knowledge.",
   "examples": [
      {"role": "user", "content": "Question: Find the person who said Taste cannot be controlled by law, what did this person die from? Reasoning path: Taste cannot be controlled by law.->media_common.quotation.author->Thomas Jefferson"}, 
      {"role": "assistant", "content": "Answer: Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said 'Taste cannot be controlled by law,' which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead."},
      {"role": "user", "content": "Question: The artist nominated for The Long Winter lived where? Reasoning path: The Long Winter->book.written_work.author->Laura Ingalls Wilder->people.person.places_lived, Unknown-Entity->people.place_lived.location, De Smet"},
      {"role": "assistant", "content": "Answer: Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in De Smet. Therefore, the answer to the question is De Smet."},
      {"role": "user", "content": "Question: Who is the coach of the team owned by Steve Bisciotti? Reasoning path: Steve Bisciotti->sports.professional_sports_team.owner_s->Baltimore Ravens->sports.sports_team_owner.teams_owned->Baltimore Ravens->organization.organization_founder.organizations_founded->Allegis Group"},
      {"role": "assistant", "content": "Answer: Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question."}
   ],
   "prompt": "Given a question and the associated retrieved reasoning path from a knowledge graph, you are asked to answer the following question based on the reasoning path and your knowledge. \nQuestion: {question} \nReasoning path: {q_entity}->{reasoning_path} \nOnly return the answer to the question."
}
   
relation_prune_prompt = {
   "system": "Given a question and the starting entity from a knowledge graph, you are asked to select the best reasoning paths from the provided candidates to find a useful reasoning path for answering the question.",
   "examples": [
      {"role": "user", "content": "Question: Name the president of the country whose main spoken language was Brahui in 1980? To proceed, the starting entity is Brahui Language Out of the following candidates for the next step, choose the best 3 with the highest probability to lead to a useful reasoning path for answering the question. Candidates: 1:language.human_language.main_country->entity; 2:language.human_language.language_family->entity; 3:language.human_language.iso_639_3_code->entity; 4:base.rosetta.languoid.parent->entity; 5:language.human_language.writing_system->entity; 6:base.rosetta.languoid.languoid_class->entity; 7:language.human_language.countries_spoken_in->entity; 8:kg.object_profile.prominent_type->entity; 9:base.rosetta.languoid.document->entity; 10:base.ontologies.ontology_instance.equivalent_instances->entity; 11:base.rosetta.languoid.local_name->entity; 12:language.human_language.region->entity Only return the index of the 3 selected reasoning path in a list."},
      {"role": "assistant", "content": "Answer: [2, 3, 4]"},
   ],
   "prompt": "Given a question and the starting entity from a knowledge graph, you are asked to select the best reasoning paths from the provided candidates to find a useful reasoning path for answering the question. \nQuestion: {question} \nTo proceed, the starting entity is {q_entity} \nFollow the planning context {plan_context} and out of the following candidates for the next step, choose the best {top_k} with the highest probability to lead to a useful reasoning path for answering the question. \nPath candidates: {path_candidates} \nOnly return the index of the {top_k} selected reasoning path in a list. \nAnswer:"
}

terminals_prune_prompt = {
   "system": "Given a question and the starting entity from a knowledge graph, you are asked to answer whether it's sufficient for you to answer the question with the following reasoning path.",
   "examples": [
      {"role": "user", "content": "Question: Find the person who said Taste cannot be controlled by law, what did this person die from? Reasoning path: Taste cannot be controlled by law.->media_common.quotation.author->Thomas Jefferson"},
      {"role": "assistant", "content": "Answer: No."},
      {"role": "user", "content": "Question: The artist nominated for The Long Winter lived where? Reasoning path: The Long Winter->film.film.film_festivals->artist->southeastern Dakota Territory"},
      {"role": "assistant", "content": "Answer: Yes."},
      {"role": "user", "content": "Question: Rift Valley Province is located in a nation that uses which form of currency? Reasoning path: Rift Valley Province->location.location.containedby->Kenya->location.country.currency_used->Kenyan Shilling (KES)"},
      {"role": "assistant", "content": "Answer: Yes."},
      {"role": "user", "content": "Question: Who is the coach of the team owned by Steve Bisciotti? Reasoning path: Steve Bisciotti->organization.organization.board_members->team->sports.sports_team_coach_tenure.coach->person"},
      {"role": "assistant", "content": "Answer: No."},
      {"role": "user", "content": "Question: The country with the National Anthem of Bolivia borders which nations? Reasoning path: National Anthem of Bolivia->music.composition.lyricist->country->location.country.borders->South America"},
      {"role": "assistant", "content": "Answer: Yes."},
   ],
   "prompt": "Given a question and the starting entity from a knowledge graph, you are asked to answer whether it's sufficient for you to answer the question with the following reasoning path. \nQuestion: {question} \nReasoning path: {q_entity}->{reasoning_path} \nIf it is sufficient to answer the question, respond with 'Yes'; otherwise, respond with 'No'. \nAnswer:"
}

beam_search_prompt = {
   "system": "Given a question and the starting entity from a knowledge graph, you are asked to retrieve reasoning paths from the given reasoning paths that are useful for answering the question.",
   "examples": [
      {"role": "user", "content": "Your goal is to retrieve 3 reasoning paths from the given reasoning paths that are useful for answering the following question. \nQuestion: What university did Barack Obama attend? \nOut of the following candidates for reasoning paths, choose the best 3 with the highest probability to lead to a useful reasoning path for answering the question. \nCandidates: \n1. Barack Obama-> spouse -> Michelle Obama -> almaMater -> Princeton University \n2. Barack Obama -> almaMater -> Harvard Law School \n3. Barack Obama -> bornIn -> Honolulu -> locatedIn -> Hawaii \n4. Barack Obama -> occupation -> President of the United States -> predecessor -> George W. Bush \n5. Barack Obama -> almaMater -> Columbia University \n6. Barack Obama -> politicalParty -> Democratic Party -> foundedIn -> 1828 \nOnly return the index of the 3 selected reasoning paths in a list."},
      {"role": "assistant", "content": "Answer: [2, 5, 1]"},
   ],
   "prompt": "Your goal is to retrieve {beam_width} reasoning paths from the given reasoning paths that is useful for answering the following question. \nQuestion: {question} \nFollow the planning context {plan_context} and out of the following candidates for reasoning paths, choose the best {beam_width} with the highest probability to lead to a useful reasoning path for answering the question. \nCandidates: {reasoning_paths} \nOnly return the index of the {beam_width} selected reasoning paths in a list."
}

plan_prompt = {
   "system": "Given a question, you are asked to generate a plan to find a reasoning path from a knowledge graph that is useful for answering the question.",
   "examples": [],
   "prompt": "Your goal is to find a reasoning path from a knowledge graph that is useful for answering the following question. {question} \nPlease generate a plan to find the reasoning path."
}

action_space = """
SEARCH(entity): out of the following candidates for the next step, choose the best top-k with the highest probabilities to lead to a useful reasoning path for answering the questions.

FORWARD(entity, relation): forward to the next entity with the given relation

BACKWARD(cur_reasoning_path): backward to the previous entity if the current reasoning path is not useful

EVALUATE[string]: answer whether it's sufficient for you to answer the question with the following reasoning path, return 'yes' if it is sufficient to answer the question, otherwise, respond with 'no'

FINISH[string]: return the final entity from the reasoning path as the final answer
"""