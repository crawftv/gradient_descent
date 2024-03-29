import dirtyjson
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama

query_str = """Claire Rose Cliteur on Instagram: "places to explore on your trip to paris | part 1 museums\xa0  1. @museerodinparis\xa0 one of the most charming museums in paris, perfect to explore on a sunlit morning. once rodin’s workshop, the museum now houses a big collection of his works. highly recommend taking a stroll in the gardens after and having a coffee there.  2. @louisvuittonfoundation an LVMH foundation which houses large modern art exhibitions by artists like rothko or basquiat, as well as a rotating permanent collection  3. @palaisdetokyo\xa0 modern art museum that always has several interesting exhibitions going on  4. @fondationcartier  contemporary art museum hosting large scale themed and commissioned exhibitions of famous and emerging artists\xa0  5. @museecarnavalet fascinating museum of the history of paris through millenia which also has an amazing restaurant in the courtyard for summer evenings with friends  6. @palaisgallieramuseedelamode museum hosting fashion exhibitions, must-visit for lovers of fashion history  7. @museeorangerie the famous gem housing monet’s water lilies. if you manage to come at a quieter time, sitting down on the bench and looking at the paintings for a while is such a serene experience\xa0  8. @museebourdelle breathtaking sculpture museum housing the works and studio of antoine bourdelle, a contemporary of rodin  9. @petitpalais_musee museum of fine art which has a vast collection of paintings, remember to check the downstairs as well, and afterwards sit down for a coffee in their beautiful courtyard\xa0  10. @boursedecommerce art space which houses the works and exhibitions part of the pinault collection in paris  and of course the louvre, musee d’orsay and versailles go without saying"""


def propositional_splitter(query_str: str):
    prompt = PromptTemplate(
        """Decompose the "Content" into clear and simple propositions, ensuring they are interpretable out of context.
    1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
    whenever possible.
    2. For any named entity that is accompanied by additional descriptive information, separate this
    information into its own distinct proposition.
    3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
    and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
    entities they refer to.
    4. Present the results as a list of strings, formatted in JSON.
    Input: Title: Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content: ¯
    The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
    1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
    other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
    frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
    origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
    that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
    both occur on grassland and are first seen in the spring. In the nineteenth century the influence
    of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
    German immigrants then exported the custom to Britain and America where it evolved into the
    Easter Bunny."
    Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
    1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
    medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
    the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
    the possible explanation for the connection between hares and the tradition during Easter", "Hares
    were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
    for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
    that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
    hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
    century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
    throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
    Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
    Britain and America." ]
    Input: {query_str}
    Output:""")
    prompt = prompt.format(query_str=query_str)
    resp = Ollama(model="mistral", temperature=0, request_timeout=500, ).complete(prompt)
    return dirtyjson.loads(resp.text)


if __name__ == "__main__":
    propositional_splitter(query_str, )
