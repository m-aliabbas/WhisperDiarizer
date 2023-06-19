import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

class SpeakerClustering(object):
    def __init__(self,num_speaker=2) -> None:
        self.num_speaker = num_speaker
        
        if self.num_speaker != 0:
            self.clustering_model = AgglomerativeClustering(self.num_speaker)
        else:
            self.clustering_model = None

    def assign_speaker_label(self,embeddings,segments):
        if self.clustering_model:
            clustering = self.clustering_model.fit(embeddings)
            labels = clustering.labels_
            print(labels)
            print(len(segments))
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)
            return segments
        else:
            score_num_speakers = {}
            for num_speakers in range(2, 10+1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
            self.num_speaker = best_num_speaker
            self.clustering_model = AgglomerativeClustering(self.num_speakers)
            self.assign_speaker_label(embeddings=embeddings,segments=segments)