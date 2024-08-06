import os

import dirtyjson
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq

accumulated_prompt = PromptTemplate("""You are an expert Q&A assistant, specializing in extracting and synthesizing the most relevant information from provided knowledge bases to answer user queries accurately and concisely.

Your role is to carefully analyze the given documents, identify the passages that directly address the user's question, and present that information in a clear, coherent response. If the knowledge base contains a complete answer, quote or paraphrase the relevant text. If it only partially answers the query, summarize the key relevant points. And if the provided documents do not contain a suitable answer, simply inform the user that an adequate answer could not be found in the given knowledge base.

Focus on extracting information solely from the provided documents, without adding any external knowledge or personal opinions. Aim to provide a response that fully addresses the query using only the most pertinent information from the knowledge base.
    -----------------
    KNOWLEDGE_BASE: {docs}
    -----------------
    QUERY: {query_str}
    -----------------
    INSTRUCTIONS:If the documents do not contain a suitable answer, simply respond: "i could not find a suitable answer".
    Do NOT suggest a solution from your own knowledge.
    Do NOT include phrases like 'Based on the provided documents, or 'According to the documents'. or  'based on the documents provided.'
    Include the url from the document metadata.
    ------------------
    Before answering the question, please think about it step-by-step within a <thinking></thinking> xml tag.
    Then, provide your final answer within <answer></answer> xml tag
    In the thinking steps, discuss the relevance of the content without mentioning document numbers.
    If you want to cite a source, consider using the author's name or the platform instead of the document number. For example: "According to @claire_rose on Instagram,..."
    EXAMPLES of  THOUGHT PROCESS:
    <examples>
    <example 1>
        Query: seafood restaurant in Paris?
        <thinking> The query asks for seafood restaurant recommendations in Paris. The provided information includes several relevant suggestions, such as a highly-regarded fish restaurant where everything is prepared exquisitely, and a top seafood spot in the city. Other details discuss wine pairings with seafood in general, which is less directly relevant to the specific query. 
        </thinking>
        <answer>
        [X] is a restaurant in Paris that features [a,b,c].
        </answer>
    </example 1>
    <example 2>
     Query: restaurant in Hong kong?
     <thinking>
     The query asks for any restaurants or dining spots in hong kong, mentioned in the documents,
     The provided documents include a recipe from a chef, John Woo, who has a restaurant in hong kong called the Hangout. The Hangout specializes in Hong Kong cuisine.
     The document also ...
     </thinking>
     <answer>
     1. The Hangout. The Hangout run by John Woo specializes in Hong Kong cuisine.
     2 ...
     </answer>
    </example 2>
    </examples>
    """)

acummulator_prompt = PromptTemplate(""" You are the world's best recommendation bot designed to answer user questions about food, travel, wine, etc.
    We have provided you a document to answer a query from an search embedding search.
    Your job is to determine whether the subject of document answers the query given the proper keywords or topics.
    if some asks for a red wine rec, only affirm red wines.
    if some asks about a city make sure the city is correct in the given document.
    QUERY: {query_str}
    ------------ Document ------------
    {doc}
    ----------------------------------
    ------------Output format------------
     Before answering the question, please think about it step-by-step within <thinking></thinking> tags.
     Then, provide your final answer within <answer>Yes/No</answer> tags.
    """)

claude_prompt = """
You are an expert Q&A assistant. Your role is to analyze the given documents, identify relevant passages, and provide a clear answer to the user's query using only information from the documents.

If the documents contain a complete answer, quote or paraphrase the relevant text. If they partially answer the query, summarize the key points. If there's no suitable answer, simply state that.

Focus solely on extracting information from the provided documents, without adding external knowledge or opinions.

KNOWLEDGE_BASE: {docs}

INSTRUCTIONS:
- If no suitable answer, respond: "I could not find a suitable answer."
- Do not suggest solutions from your own knowledge.
- Do not include phrases like "Based on the documents" or "According to the documents."

Before answering, outline your thought process within <thinking></thinking> tags.
Then, provide your final answer within <answer></answer> tags.
In the thought process, discuss content relevance without mentioning document numbers. Use author names or platforms instead, e.g., "According to @claire_rose on Instagram,..."
"""


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
    ALWAYS GIVE BACK A JSON LIST
    Input: {query_str}
    Output:""")
    prompt = prompt.format(query_str=query_str)
    resp = Groq(model="mixtral-8x7b-32768", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1).complete(prompt)
    text = resp.text
    text = text.replace("\\_", "_")
    return dirtyjson.loads(text)
