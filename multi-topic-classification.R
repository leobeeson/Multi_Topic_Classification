library(quanteda)
library(dplyr)
library(text2vec)
library(stringr)


##### LOAD MWE DICTIONARY #####
mwe <- readRDS(file = "mwe/mwe_for_compounding.RDS")

##### LOAD CLASS DOCS TOKENS OBJECT #####
tokens_class_docs <- readRDS(file = "class_docs_tokens_object.RDS")  ######################### NEXT UPDATE

##### LOAD STOPWORDS #####
stop_words <- stopwords(source = "smart")

##### LOAD MODELS #####
vectorizer <- readRDS(file = "models/vectorizer.RDS")
tfidf_model = readRDS(file = "models/tfidf_model.RDS")
lsa_tfidf_model =readRDS(file = "models/lsa_tfidf_model.RDS")
lda_model = readRDS(file = "models/lda_model.RDS")

##### READ IN TEST DATA #####
sentences_with_IDs <- read.csv(file = "mwe/corpus_with_IDs.csv", header = FALSE, sep = "\t", stringsAsFactors = FALSE)
colnames(sentences_with_IDs) <- c("ID", "text")

##### TOKENIZE TEST DATA #####
corpus_all <- corpus(sentences_with_IDs, text_field = "text", docid_field = "ID")

tokens_obj <- tokens_tolower(tokens(x = corpus_all, remove_hyphens = FALSE, remove_punct = TRUE, remove_numbers = TRUE, remove_symbols = TRUE))
tokens_obj_1 <- tokens_select(tokens_obj, pattern = "^[a-z-]+$", valuetype = "regex", selection = "keep", padding = TRUE)
tokens_obj_2 <- tokens_compound(x = tokens_obj_1, pattern = phrase(mwe), valuetype = "fixed", concatenator = "_")
tokens_obj_3 <- tokens_select(x = tokens_obj_2, pattern = "\\w{3,}", valuetype = "regex", case_insensitive = FALSE)
tokens_clusters <- tokens_remove(tokens_obj_3, stop_words)
#tokens_clusters <- tokens_wordstem(tokens_obj_4, language = quanteda_options("language_stemmer"))
rm(tokens_obj, tokens_obj_1, tokens_obj_2, tokens_obj_3)

##################################     ########################
##### CALCULATE SIMILARITIES #####     ##### BY SENTENCES #####
##################################     ########################

### SELECTED HYPERPARAMETERS
min_word_freq <- 20
min_doc_freq <- 5
top_n <- 5
k_lsa <- 50
k_lda <- 50

##### MERGE TOKENS OBJECTS FROM article_dfS #####
tokens_all <- append(tokens_clusters, tokens_class_docs)

##### CREATE DTMS #####
it_clusters = itoken(as.list(tokens_clusters))
it_class_docs = itoken(as.list(tokens_class_docs))
it_all = itoken(as.list(tokens_all))

dtm_clusters = create_dtm(it_clusters, vectorizer)
dtm_class_docs = create_dtm(it_class_docs, vectorizer)
dtm_all = create_dtm(it_all, vectorizer)

scaling_vector <- 1

predict_top_n_tags <- function(doc_idx, n, sim_matrix, scaling_vector,
                               col_start = 1, col_end = ncol(sim_matrix)){
  doc_row <- sim_matrix[doc_idx, col_start:col_end]
  doc_row_scaled <- scale_classes(doc_row, scaling_vector)
  index_top_n <- order(doc_row_scaled, decreasing = TRUE)[1:n]
  predicted_tags <- doc_row_scaled[index_top_n]
  predicted_tags <- predicted_tags[ifelse(predicted_tags > 0, TRUE, FALSE)]
  cluster_num = rownames(sim_matrix)[doc_idx]
  predicted_tag_names <- names(predicted_tags)
  output <- list(cluster = cluster_num, 
                 predicted_tags = predicted_tags, 
                 predicted_tag_names = predicted_tag_names)
  return(output)
}

scale_classes <- function(doc_row, scaling_vector){
  doc_row_scaled <- doc_row * scaling_vector
  return(doc_row_scaled)
}

model_logger <- data.frame(cluster = character(),
                           method = character(),
                           predicted_classes = character(),
                           predicted_classes_values = character(),
                           predicted_classes_names = character(),
                           stringsAsFactors = FALSE)

##### GET JACCARD SIMILARITY #####
sim_jac = sim2(dtm_clusters, dtm_class_docs, method = "jaccard", norm = "none")
for(i in seq(from = 1, to = dtm_clusters@Dim[1], by = 1)){
  check_sim <- predict_top_n_tags(doc_idx = i, n = top_n, sim_matrix = sim_jac, 
                                  scaling_vector = scaling_vector)
  temp <- list(cluster = check_sim$cluster,
               method = "jaccard",
               predicted_classes = paste(names(check_sim$predicted_tags), collapse = "\n "),
               predicted_classes_values = paste(check_sim$predicted_tags, collapse = "\n "),
               predicted_classes_names = paste(check_sim$predicted_tag_names, collapse = "\n "))
  model_logger <- rbind(model_logger, temp, stringsAsFactors = FALSE)
}
rm(sim_jac)

##### GET COSINE SIMILARITY #####
sim_cos = sim2(dtm_clusters, dtm_class_docs, method = "cosine", norm = "l2")
for(i in seq(from = 1, to = dtm_clusters@Dim[1], by = 1)){
  check_sim <- predict_top_n_tags(doc_idx = i, n = top_n, sim_matrix = sim_cos, 
                                  scaling_vector = scaling_vector)
  temp <- list(cluster = check_sim$cluster,
               method = "cosine",
               predicted_classes = paste(names(check_sim$predicted_tags), collapse = "\n "),
               predicted_classes_values = paste(check_sim$predicted_tags, collapse = "\n "),
               predicted_classes_names = paste(check_sim$predicted_tag_names, collapse = "\n "))
  model_logger <- rbind(model_logger, temp, stringsAsFactors = FALSE)
}
rm(sim_cos)

##### GET COSINE SIMILARITY WITH TF-IDF#####
dtm_tfidf = transform(dtm_all, tfidf_model)
sim_cos_tfidf = sim2(x = dtm_tfidf, method = "cosine", norm = "l2")
for(i in seq(from = 1, to = dtm_clusters@Dim[1], by = 1)){
  check_sim <- predict_top_n_tags(doc_idx = i, n = top_n, sim_matrix = sim_cos_tfidf, 
                                  scaling_vector = scaling_vector,
                                  col_start = dtm_clusters@Dim[1] + 1, col_end = dtm_all@Dim[1])
  temp <- list(cluster = check_sim$cluster,
               method = "cos-tfidf",
               predicted_classes = paste(names(check_sim$predicted_tags), collapse = "\n "),
               predicted_classes_values = paste(check_sim$predicted_tags, collapse = "\n "),
               predicted_classes_names = paste(check_sim$predicted_tag_names, collapse = "\n "))
  model_logger <- rbind(model_logger, temp, stringsAsFactors = FALSE)
}
rm(sim_cos_tfidf)

##### GET COSINE SIMILARITY WITH LSA WITH TF-IDF#####
# dtm_tfidf_lsa = transform(dtm_tfidf, lsa_tfidf_model)
# sim_lsa_tfidf = sim2(x = dtm_tfidf_lsa, method = "cosine", norm = "l2")
# for(i in seq(from = 1, to = dtm_clusters@Dim[1], by = 1)){
#   check_sim <- predict_top_n_tags(doc_idx = i, n = top_n, sim_matrix = sim_lsa_tfidf,
#                                   scaling_vector = scaling_vector,
#                                   col_start = dtm_clusters@Dim[1] + 1, col_end = dtm_all@Dim[1])
#   temp <- list(cluster = check_sim$cluster,
#                method = "lsa-tfidf",
#                predicted_classes = paste(names(check_sim$predicted_tags), collapse = "\n "),
#                predicted_classes_values = paste(check_sim$predicted_tags, collapse = "\n "),
#                predicted_classes_names = paste(check_sim$predicted_tag_names, collapse = "\n "))
#   model_logger <- rbind(model_logger, temp, stringsAsFactors = FALSE)
# }

##### GET COSINE SIMILARITY WITH LDA #####
# doc_topic_distr = lda_model$transform(dtm_all)
# sim_lda = sim2(x = doc_topic_distr, method = "cosine", norm = "l2")
# for(i in seq(from = 1, to = dtm_clusters@Dim[1], by = 1)){
#   check_sim <- predict_top_n_tags(doc_idx = i, n = top_n, sim_matrix = sim_lda,
#                                   scaling_vector = scaling_vector,
#                                   col_start = dtm_clusters@Dim[1] + 1, col_end = dtm_all@Dim[1])
#   temp <- list(cluster = check_sim$cluster,
#                method = "lda",
#                predicted_classes = paste(names(check_sim$predicted_tags), collapse = "\n "),
#                predicted_classes_values = paste(check_sim$predicted_tags, collapse = "\n "),
#                predicted_classes_names = paste(check_sim$predicted_tag_names, collapse = "\n "))
#   model_logger <- rbind(model_logger, temp, stringsAsFactors = FALSE)
# }

##### INITIALISE PREDICTION OBJECT #####
crowded_prediction_sent <- data.frame(sentence = character(),
                                      predicted_classes = character(),
                                      predicted_classes_values = character(),
                                      stringsAsFactors = FALSE)

##### CALCULATE AVERAGE DOCUMENT LENGTH FOR PIVOTED LENGTH NORMALIZATION MEASURE #####
method_weights <- c(1, 1, 1)#, 0.01, 0.01)
names(method_weights) <- c("jaccard", "cosine", "cos-tfidf") #, "lda", "lsa-tfidf")
pln_param <- 1
selection_threshold <- 0.001

mean_doc_length <- mean(unlist(lapply(tokens_clusters, function(x) length(x))))
pln <- 1 - pln_param + pln_param * unlist(lapply(tokens_clusters, function(x) length(x)))/mean_doc_length

##### CROWD PREDICTIONS #####
for(doc in unique(model_logger$cluster)){
  
  methods <- model_logger[model_logger$cluster == doc, ]
  cluster <- methods$cluster[1]
  index <- rep(0, length(tokens_class_docs))
  names(index) <- names(tokens_class_docs)
  
  for(method in unique(methods$method)){

    tags <- methods$predicted_classes[methods$method == method]
    tags <- unlist(str_split(tags, pattern = "\n "))
    scores <- methods$predicted_classes_values[methods$method == method]
    scores <- as.numeric(unlist(str_split(scores, pattern = "\n ")))
    scores <- scores / sum(scores)
    names(scores) <- tags
    weighted_scores <- scores * method_weights[method]
    index[names(scores)] <- index[names(scores)] + scores
  }
  index <- index / length(method_weights)
  #predicted_tags <- diversify_level_2_jel_codes(index, input$max_level_2_tags, max_number_of_predicted_tags)
  ordered_weighed_index <- index[order(index, decreasing = TRUE)]
  predicted_tags <- ordered_weighed_index  * pln[doc]
  
  selected_JEL_codes <- predicted_tags[predicted_tags >= selection_threshold]
  
  temp <- data.frame(sentence = doc,
                     predicted_clases = paste(names(selected_JEL_codes), collapse = ", "),
                     predicted_clases_values = paste(selected_JEL_codes, collapse = ", "),
                     stringsAsFactors = FALSE)
  crowded_prediction_sent <- rbind(crowded_prediction_sent, temp, stringsAsFactors = FALSE)
}

write.csv(crowded_prediction_sent, file = "Topics_per_Document.csv")
