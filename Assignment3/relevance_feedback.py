import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def relevance_feedback(vec_docs, vec_queries, sim, gt, n=10):
    """
    relevance feedback
    Parameters
        ----------
        
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    alpha = 0.7
    beta = 0.3
    N = n
    iters = 10

    queries = vec_queries.toarray()
    docs = vec_docs.toarray()

    rf_sim = sim

    for nq, q in enumerate(queries):

        for i in range(iters):

            q_new = queries[nq]

            ranked_documents = np.argsort(-rf_sim[:, nq])
            top_N = ranked_documents[:N]

            relevant_sum = np.zeros((1, 10625))
            non_relevant_sum = np.zeros((1, 10625))

            relevant_gt_docs = []

            for g in gt:
                if g[0] == (nq + 1):
                    relevant_gt_docs.append(g[1]-1)

            for cur_doc in top_N:
                if cur_doc in relevant_gt_docs:
                    relevant_sum += docs[cur_doc]
                else:
                    non_relevant_sum = docs[cur_doc]            

            q_new = q_new + alpha * relevant_sum - beta * non_relevant_sum

            queries[nq] = q_new

    rf_sim = cosine_similarity(docs, queries)
        

    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, gt, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """

    rf_sim = sim  # change
    
    alpha = 0.7
    beta = 0.3
    N = n
    iters = 10

    queries = vec_queries.toarray()
    docs = vec_docs.toarray()

    rf_sim = sim

    for nq, q in enumerate(queries):

        for it in range(iters):
            q_new = queries[nq]

            ranked_documents = np.argsort(-rf_sim[:, nq])
            top_N = ranked_documents[:N]

            relevant_sum = np.zeros((1, 10625))
            non_relevant_sum = np.zeros((1, 10625))

            relevant_gt_docs = []

            for g in gt:
                if g[0] == (nq + 1):
                    relevant_gt_docs.append(g[1]-1)

            for cur_doc in top_N:
                if cur_doc in relevant_gt_docs:
                    relevant_sum += docs[cur_doc]
                else:
                    non_relevant_sum = docs[cur_doc]            

            q_new = q_new + alpha * relevant_sum - beta * non_relevant_sum

            top_words_all = []
            
            for rel_doc in relevant_gt_docs:
                vec_doc = docs[rel_doc]
                vec_doc_sorted_args = np.argsort(-vec_doc)
                vec_doc_sorted = np.sort(-vec_doc)
                
                for i in range(N):
                    top_words_all.append((-vec_doc_sorted[i], vec_doc_sorted_args[i]))
                    
            top_words_all_sorted = sorted(top_words_all, reverse= True)[:N]

            to_add = np.zeros((1,10625))

            for wrd in top_words_all_sorted:    
                to_add[0, wrd[1]] += wrd[0]

            q_new +=  to_add
            queries[nq] = q_new

    rf_sim = cosine_similarity(docs, queries)

    return rf_sim
