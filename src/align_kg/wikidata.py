from collections import defaultdict
import os
import pickle

import requests

class WikidataCache:
   def __init__(self, cache_f):
      self.cache_f = cache_f 

      if os.path.exists(cache_f):
         with open(cache_f, "rb") as f:
               cache = pickle.load(f)
         
         self.qid = cache["qid"]
         self.triples = cache["triples"]
      else:
         self.qid = {}
         self.triples = {}
   
   def add_triples(self, entity, triples):
      self.triples[entity] = triples 
   
   def add_qids(self, qids):
      if qids:
         self.qid.update(qids)
   
   def save_cache(self):
      cache = {
         "qid": self.qid,
         "triples": self.triples
      }

      with open(self.cache_f, "wb") as f:
         pickle.dump(cache, f)


class Triple:
   def __init__(self, head, relation, tail):
      self.triple = (head, relation, tail)
      self.qualifiers = defaultdict(list)
   
   def add_qualifier(self, relation, tail):
      self.qualifiers[relation].append(tail)
   
   def __str__(self):
      triple = f'({self.triple[0]}, {self.triple[1]}, {self.triple[2]})'

      if len(self.qualifiers) == 0:
         return triple
      
      qualifiers = []
      for qualifier_relation, qualifier_values in self.qualifiers.items():
         s = ", ".join(qualifier_values)

         if len(qualifier_values) > 1:
               s = f"[{s}]"
               
         s = f"{qualifier_relation}: {s}"
         
         qualifiers.append(s)
      
      qualifiers = "; ".join(qualifiers)
      qualifiers = f"{{{qualifiers}}}"

      s = f"({triple}, {qualifiers})"
      return s


class Wikidata:

   def get_triples(self, entity, is_label):
      """
      Args:
         entity (string): Entity label or QID.
         is_label (bool): True if entity is a label not a QID 
      
      Returns:
         tuple: (triples, tail_qids)

         triples is a list of the triples. 
               Each triple is a string in the following format:
               ((head, relation, tail), {qualifier_relation: qualifier_tail, etc.})

         tail_qids is a dict mapping tail labels to their QIDs. 
      """
      if is_label:
         qid = self._label_to_id(entity)
      else:
         qid = entity
      
      triples, tail_qids = self._get_triples_from_qid(qid)
      return triples, tail_qids

   def get_description(self, entity, is_label):
      if is_label:
         qid = self._label_to_id(entity)
      else:
         qid = entity 
      
      response = self._get_entity(qid)

      if not response:
         return

      try:
         entity_description = response["entities"][qid]["descriptions"]["en"]["value"]
         return entity_description
      except:
         return  
   
   def triples_batch_iterator(self, triples):
      """
      Batch triples such that all triples with the same relation will be in the same batch
      """
      batch_size = 15

      d = defaultdict(list)
      for triple in triples:
         relation = triple.triple[1]
         d[relation].append(triple)
      
      cur_batch = []
      for relation, relation_triples in d.items():
         if len(cur_batch) + len(relation_triples) <= batch_size:
               cur_batch += relation_triples
         else:
               yield self.verbalize_triples(cur_batch)
               cur_batch = relation_triples
      
      if cur_batch:
         yield self.verbalize_triples(cur_batch)
      

   def _get_triples_from_qid(self, qid):
      response = self._get_entity(qid)

      if not response:
         return None, None

      triples = []
      tail_qids = {} 

      entity_info = response["entities"][qid]

      try:
         entity_label = entity_info["labels"]["en"]["value"]
      except:
         return None, None

      try:
         entity_description = entity_info["descriptions"]["en"]["value"]
         triple = Triple(entity_label, "is", entity_description)
         triples.append(triple)
      except:
         pass 

      entity_claims = entity_info["claims"]

      for property_id, statements in entity_claims.items():
         property_label = self._id_to_label(property_id)

         for statement in statements:
               mainsnak = statement["mainsnak"]
               if mainsnak["snaktype"] != "value":
                  continue
               
               if mainsnak["datatype"] == "wikibase-item":
                  tail_id, tail_label = self._get_snak_value(mainsnak)
                  mainsnak_value = tail_label
               else:
                  mainsnak_value = self._get_snak_value(mainsnak)
               
               if not mainsnak_value:
                  continue
               
               if mainsnak["datatype"] == "wikibase-item":
                  tail_qids[tail_label] = tail_id

               triple = Triple(entity_label, property_label, mainsnak_value)

               # qualifiers
               if not "qualifiers" in statement:
                  triples.append(triple)
                  continue

               qualifiers = statement["qualifiers"]

               for qualifier_property_id, qualifier_snaks in qualifiers.items():
                  qualifier_property_label = self._id_to_label(qualifier_property_id)

                  for qualifier_snak in qualifier_snaks:
                     if qualifier_snak["snaktype"] != "value":
                           continue

                     if qualifier_snak["datatype"] == "wikibase-item":
                           qualifier_tail_id, qualifier_tail_label = self._get_snak_value(qualifier_snak)
                           qualifier_snak_value = qualifier_tail_label
                     else:
                           qualifier_snak_value = self._get_snak_value(qualifier_snak)
                           
                     if not qualifier_snak_value:
                           continue

                     if qualifier_snak["datatype"] == "wikibase-item":
                           tail_qids[qualifier_tail_label] = qualifier_tail_id

                     triple.add_qualifier(qualifier_property_label, qualifier_snak_value)

               triples.append(triple)
      
      # triples = self.verbalize_triples(triples)
               
      return triples, tail_qids

   def verbalize_triples(self, triples):
      return [str(triple) for triple in triples]

   
   def _valid_response(self, response):
      return response.status_code == 200

   def _search_label(self, label):
      url = "https://www.wikidata.org/w/api.php"
      params = {
         "action": "wbsearchentities",
         "format": "json",
         "language": "en",
         "search": label,
      }

      response = requests.get(url, params=params)
      if self._valid_response(response):
         return response.json()

   def _get_entity(self, id):
      url = f"https://www.wikidata.org/wiki/Special:EntityData/{id}.json"

      response = requests.get(url)
      if self._valid_response(response):
         return response.json()

   def _id_to_label(self, id):
      response = self._get_entity(id)
      if response:
         try:
               return response["entities"][id]["labels"]["en"]["value"]
         except:
               return None
   
   def _label_to_id(self, label):
      response = self._search_label(label)
      if not response:
         return

      search_results = response["search"]
      if not search_results:
         return 

      qid = search_results[0]["id"]  # TODO: assumes first

      return qid


   def _get_snak_value(self, snak):
      try:
         datatype = snak["datatype"]
         datavalue = snak["datavalue"]["value"]
      except:
         return None

      if datatype == "wikibase-item":
         tail_id = datavalue["id"]
         tail_label = self._id_to_label(tail_id)
         snak_value = (tail_id, tail_label)
         
      elif datatype == "time":
         snak_value = datavalue["time"]
         
      elif datatype == "string":
         snak_value = datavalue
         
      elif datatype == "quantity":
         amount = datavalue["amount"]
         unit = datavalue["unit"]
         
         if unit == "1":
               snak_value = amount
         elif "wikidata.org/entity" in unit:
               unit_id = unit.split("/")[-1]
               unit_label = self._id_to_label(unit_id)
               snak_value = amount + " " + unit_label
               
      elif datatype == "globe-coordinate":
         latitude = datavalue["latitude"]
         longitude = datavalue["longitude"]
         snak_value = f"({latitude}, {longitude})"
         
      elif datatype == "math":
         snak_value = datavalue
         
      else:
         snak_value = None
      
      return snak_value
   
      # match datatype:
      #    case "wikibase-item":
      #          tail_id = datavalue["id"]
      #          tail_label = self._id_to_label(tail_id)
      #          snak_value = (tail_id, tail_label)

      #    case "time":
      #          snak_value = datavalue["time"]

      #    case "string":
      #          snak_value = datavalue

      #    case "quantity":
      #          amount = datavalue["amount"]
      #          unit = datavalue["unit"]

      #          if unit == "1":
      #             snak_value = amount
      #          elif "wikidata.org/entity" in unit:
      #             unit_id = unit.split("/")[-1]
      #             unit_label = self._id_to_label(unit_id)
      #             snak_value = amount + " " + unit_label

      #    case "globe-coordinate":
      #          latitude = datavalue["latitude"]
      #          longitude = datavalue["longitude"]
      #          snak_value = f"({latitude}, {longitude})"

      #    case "math":
      #          snak_value = datavalue

      #    case _:
      #          snak_value = None

def main():
   wikidata = Wikidata()
   print(wikidata.get_description("Q42", False))
   triples, tail_qids = wikidata.get_triples("Q42", False)
   print(triples, tail_qids)

if __name__ == "__main__":
   main()