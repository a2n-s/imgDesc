# 2022 Supaero Data Science MS Workshop

## Table Of Content.
- **1.** [**Assignment**](#1-assignment-toc)
- **1.1.** [**Principle and pedagogical goal**](#11-principle-and-pedagogical-goal-toc)
- **1.2.** [**Orders and evaluation criteria**](#12-orders-and-evaluation-criteria-toc)
- **1.3.** [**Evaluation**](#13-evaluation-toc)
- **1.4.** [**Recommandations**](#14-recommandations-toc)
- **2.** [**My subject**](#2-my-subject-toc)

## 1. Assignment [[toc](#table-of-content)]
### 1.1. Principle and pedagogical goal [[toc](#table-of-content)]
Le principe est celui de la co-évaluation et le but est que le temps que vous consacrez à l'évaluation soit un temps où vous continuez à découvrir des choses nouvelles et à approfondir votre maîtrise en ML.  
Le fichier en pièce jointe vous indique, pour chaque étudiant.e, un sujet à traiter. Le but de l'exercice est de rédiger un notebook présentant le sujet, comme si vous le présentiez à vos pairs (étudiant SDD, collègue de travail, client compétent sur le sujet, etc.). On n'apprend jamais aussi bien que quand on explique, c'est donc l'occasion de bien maîtriser un sujet de plus et de monter en compétence collectivement en portant un regard critique sur nos productions respectives.

### 1.2. Orders and evaluation criteria [[toc](#table-of-content)]
Votre notebook doit être didactique, agréable à lire, avoir un bon équilibre entre aspects formels et pratiques. Il doit être jouable en environ une heure. Selon les sujets, vous aurez plus ou moins à faire d'efforts pour illustrer le sujet en pratique ou à rendre la théorie accessible. La langue de rédaction est le français ou l'anglais, selon votre préférence (et la qualité de la langue contribue à des notebooks agréables à lire).  
Votre notebook doit être rigoureux : ce n'est pas de la vulgarisation scientifique, vous vous devez d'être précis et rigoureux. Ca ne vous oblige pas à rédiger des preuves mathématiques mais ça nécessite de formuler et de discuter des idées et résultats de façon précise et argumentée.  
Votre notebook doit également être utile et réutilisable, comporter des éléments (dessins, code, texte) qui permettront au lecteur d'être rapidement fonctionnel sur le sujet.  
Votre notebook doit être documenté : les aspects non-abordés ou les extensions peuvent pointer vers des ressources en ligne ou des éléments bibliographiques, les idées avancées doivent être soutenues par des références. Vous pouvez joindre des annexes.  
Important : votre notebook doit être anonyme (le non respect de cette consigne est éliminatoire).  
Les notebooks devront être soumis sur ce site https://openreview.net/group?id=supaerodatascience.github.io/SupaeroSDD/2022/Workshop avant la date limite de soumission (18 janvier, 01:00 GMT, aucune extension deadline ne sera accordée et le non-rendu est éliminatoire). C'est sur ce site que se déroulera ensuite le processus de co-évaluation (plus d'infos sur cela bientôt).  

### 1.3. Evaluation [[toc](#table-of-content)]
Le 18/01, nous avons une séance où nous jouerons les notebooks. Le but sera d'évaluer chaque notebook, en binôme, pendant la première heure, d'avoir le temps de rédiger une évaluation (que vous pourrez éventuellement corriger plus tard), de prendre une pause, puis de recommencer avec un second notebook. Les binômes seront différents pour le premier et le second notebook. Jouer et noter les notebooks en binôme est important car cela vous permet d'en discuter au fil du notebook. Les évaluations seront constituées de notes numériques et d'éléments textuels où le binôme évaluateur devra résumer sa compréhension du notebook et argumenter sur les points forts et les points faibles.  
A l'issue de cette séance, chaque notebook aura donc reçu une évaluation (descernée par un binôme) et chaque binôme aura évalué (donc découvert) deux notebooks (donc deux nouveaux sujets).  
Je vous demanderai ensuite de répéter cet exercice à la maison pour deux notebooks supplémentaires. Les premières évaluations resteront confidentielles à ce stade. Les évaluations correspondantes seront à rendre pour le 23/01. Chaque notebook aura donc deux évaluations et, au total, vous aurez chacun.e découvert 5 sujets (en comptant celui que vous aurez rédigé).  
Par ailleurs, chaque notebook recevra (au moins) une évaluation d'un correcteur externe.  
A la fin, je compilerai toutes les évaluations et notes pour en tirer une évaluation unifiée.  
Dans l'hypothèse où nous ne pourrons pas nous réunir en présentiel, la séance du 18/01 demeurera dédiée aux notebooks mais je vous demanderai de les évaluer depuis chez vous, toujours en binôme, en utilisant la visio si nécessaire.  

### 1.4. Recommandations [[toc](#table-of-content)]
La rédaction d'un notebook prend du temps, mais c'est aussi une des meilleures manières d'apprendre en profondeur. Voici un petit timing type que je vous recommande, lissé sur 6 fois 2h de travail.  
Séance 1 : recherche de sources et lecture efficace. Chaque notebook dispose d'une indication bibliographique, à vous de la lire efficacement et de chercher des sources complémentaires pour mieux comprendre ou apporter un éclairage différent.  
Séance 2 : lecture approfondie et expérience pratique du sujet (code, exploration personnelle de la théorie). Cette séance demandera peut-être à être doublée.  
Séance 3 : décision sur la trame du notebook et ébauche de rédaction.  
Séance 4 : rédaction.  
Séance 5 : rédaction.  
Séance 6 : relecture et corrections.  

## 2. My subject [[toc](#table-of-content)]
Deep Visual-Semantic Alignments for Generating Image Descriptions:
- the [scholar portal][karpathy2015deep-portal]
- the [paper][karpathy2015deep]
- the [standford post][karpathy2015deep-blog]
- the [code][karpathy2015deep-code]

### 2.1. Abstract
We present a model that generates natural language de-
scriptions of images and their regions. Our approach lever-
ages datasets of images and their sentence descriptions to
learn about the inter-modal correspondences between lan-
guage and visual data. Our alignment model is based on a
novel combination of Convolutional Neural Networks over
image regions, bidirectional Recurrent Neural Networks
over sentences, and a structured objective that aligns the
two modalities through a multimodal embedding. We then
describe a Multimodal Recurrent Neural Network architec-
ture that uses the inferred alignments to learn to generate
novel descriptions of image regions. We demonstrate that
our alignment model produces state of the art results in re-
trieval experiments on Flickr8K, Flickr30K and MSCOCO
datasets. We then show that the generated descriptions sig-
nificantly outperform retrieval baselines on both full images
and on a new dataset of region-level annotations.

### 2.2. Introduction
The contributions:  
• We develop a deep neural network model that in-
  fers the latent alignment between segments of sen-
  tences and the region of the image that they describe.
  Our model associates the two modalities through a
  common, multimodal embedding space and a struc-
  tured objective. We validate the effectiveness of this
  approach on image-sentence retrieval experiments in
  which we surpass the state-of-the-art.
• We introduce a multimodal Recurrent Neural Network
  architecture that takes an input image and generates
  its description in text. Our experiments show that the
  generated sentences significantly outperform retrieval-
  based baselines, and produce sensible qualitative pre-
  dictions. We then train the model on the inferred cor-
  respondences and evaluate its performance on a new
  dataset of region-level annotations.

### 2.3. Related Work
**Dense image annotations**.  
Our work shares the high-level goal of densely annotating the contents of images with many works before us.  
[\[2\]][2] [\[48\]][48] the multimodal correspondence between words and images to annotate segments of images.  
[\[34\]][34] [\[18\]][18] [\[15\]][15] [\[33\]][33] the problem of holistic scene understanding in which the scene type, objects and their spatial support in the image is inferred.  
However, the focus of these works is on correctly labeling scenes, objects and regions with a fixed set of categories, while our focus is on richer and higher-level descriptions of regions.  

**Generating descriptions**.  
[\[21\]][21] [\[49\]][49] [\[13\]][13] [\[43\]][43] [\[23\]][23] task as a retrieval problem where the most compatible annotation in the training set is transferred to a test image  
[\[30\]][30] [\[35\]][35] [\[31\]][31] task as a retrieval problem where training annotations are broken up and stitched together   

[\[19\]][19] [\[29\]][29] [\[13\]][13] [\[55\]][55] [\[56\]][56] [\[9\]][9] [\[1\]][1] generate image captions based on fixed templates that are filled based on the content of the image  
[\[42\]][42] [\[57\]][57] generate image captions based on fixed templates that are filled based on generative grammars,  
variety of possible outputs is limited.  

[\[26\]][26] log-bilinear model that can generate full sentence descriptions for images, but their model uses a fixed window context while our Recurrent Neural Network (RNN) model conditions the probability distribution over the next word in a sentence on all previously generated words.  
[\[38\]][38] [\[54\]][54] [\[8\]][8] [\[25\]][25] [\[12\]][12] [\[5\]][5] other using RNNs to generate image descriptions.  
  
Ours simpler but suffers in performance.  

**Grounding natural language in images**.  
[\[27\]][27] [\[39\]][39] [\[60\]][60] [\[36\]][36] grounding text in the visual domain.  
[\[16\]][16] associate words and images through a semantic embedding.  
[\[24\]][24] decompose images and sentences into fragments and infer their inter-modal alignment using a ranking objective. Grounding dependency tree relations,  
our model aligns contiguous segments of sentences which are more meaningful, interpretable, and not fixed in length.  

**Neural networks in visual and language domains**.  
representing images and words in higher-level representations.  
[\[32\]][32] [\[28\]][28] Convolutional Neural Networks (CNNs) have recently emerged as a powerful class of models for image classification and object detection [\[45\]][45] .  
[\[41\]][41] [\[22\]][22] [\[3\]][3] pretrained word vectors to obtain low-dimensional representations of words.  
[\[40\]][40] [\[50\]][50] language modeling, but we additionally condition these models on images.  

[karpathy2015deep-portal]:   https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Deep+Visual-Semantic+Alignments+for+Generating+Image+Descriptions&btnG= 
[karpathy2015deep]:          https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf 
[karpathy2015deep-blog]:     https://cs.stanford.edu/people/karpathy/deepimagesent/ 
[karpathy2015deep-code]:     https://github.com/karpathy/neuraltalk2
[karpathy2015deep-codedep]:  https://github.com/karpathy/neuraltalk
[karpathy2015deep-tmpvideo]: https://www.youtube.com/watch?v=e-WB4lfg30M

<!-- all the references from the paper -->
[1]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Video+in+sentences+out.&btnG=
[2]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Matching+words+and+pictures.&btnG=
[3]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Neural+probabilistic+language+models.&btnG=
[4]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Microsoft+coco+captions:+Data+collection+and+evaluation+server.&btnG=
[5]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Learning+a+recurrent+visual+representation+for+image+caption+generation.&btnG=
[6]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Imagenet:+A+large-scale+hierarchical+image+database.&btnG=
[7]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Language-specific+translation+evaluation+for+any+target+language.&btnG=
[8]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Long-term+recurrent+convolutional+networks+for+visual+recognition+and+description.&btnG=
[9]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Image+description+using+visual+dependency+representations.&btnG=
[10]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Finding+structure+in+time.&btnG=
[11]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=The+pascal+visual+object+classes+(voc)+challenge.&btnG=
[12]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=From+captions+to+visual+concepts+and+back.&btnG=
[13]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Every+picture+tells+a+story:+Generating+sentences+from+images.&btnG=
[14]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=What+do+weperceive+in+a+glance+of+a+real-world+scene?&btnG=
[15]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=A+sentence+is+worth+athousand+pixels.&btnG=
[16]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Devise:+A+deep+visual-semantic+embedding+model.&btnG=
[17]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Rich+feature+hierarchies+for+accurate+object+detection+and+semantic+segmentation.&btnG=
[18]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Decomposing+a+scene+into+geometric+and+semantically+consistent+regions.&btnG=
[19]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=From+image+annotation+to+image+description.&btnG=
[20]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Long+short-term+memory.Neural+computation,&btnG=
[21]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Framing+image+description+as+a+ranking+task:+data,+models+and+evaluation+metrics.&btnG=
[22]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Glove:+Global+vectors+for+word+representation.&btnG=
[23]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Learning+cross-modality+similarity+for+multinomial+data.&btnG=
[24]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Deep+fragment+embeddings+for+bidirectional+image+sentence+mapping.&btnG=
[25]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Unifying+visual-semantic+embeddings+with+multimodal+neural+language+models.&btnG=
[26]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Multimodal+neural+language+models.&btnG=
[27]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=What+are+you+talking+about?+text-to-image+coreference.&btnG=
[28]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Imagenet+classification+with+deep+convolutional+neural+networks.&btnG=
[29]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Baby+talk:+Understanding+and+generating+simple+image+descriptions.&btnG=
[30]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Collective+generation+of+natural+image+descriptions.&btnG=
[31]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Composition+and+compression+of+trees+for+image+descriptions.&btnG=
[32]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Gradient-based+learning+applied+to+document+recognition.&btnG=
[33]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=What,+where+and+who?+classifying+events+by+scene+and+object+recognition.&btnG=
[34]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Towards+total+scene+understanding:+Classification,+annotation+and+segmentation+in+an+automatic+framework.&btnG=
[35]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Composing+simple+image+descriptions+using+webscale+n-grams.&btnG=
[36]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Visual+semanticsearch:+Retrieving+videos+via+complex+textual+queries.&btnG=
[37]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Microsoft+coco:+Common+objects+in+context.&btnG=
[38]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Explain+images+with+multimodal+recurrent+neural+networks.&btnG=
[39]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Fox.+A+Joint+Model+of+Language+and+Perception+for+Grounded+Attribute+Learning.&btnG=
[40]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Recurrent+neural+network+based+language+model.&btnG=
[41]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Distributed+representations+of+words+and+phrases+and+their+compositionality.&btnG=
[42]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Generating+image+descriptions+from+computer+vision+detections.&btnG=
[43]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Im2text:+Describing+images+using+1+million+captioned+photographs.&btnG=
[44]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Bleu:+a+method+for+automatic+evaluation+of+machine+translation.&btnG=
[45]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Imagenet+large+scale+visual+recognition+challenge,&btnG=
[46]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Bidirectional+recurrent+neuralnetworks.&btnG=
[47]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Very+deep+convolutional+networks+for+large-scale+image+recognition.&btnG=
[48]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Connecting+modalities:+Semi-supervised+segmentation+and+annotation+of+images+using+unaligned+text+corpora.&btnG=
[49]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Grounded+compositional+semantics+for+finding+and+describing+images+with+sentences.&btnG=
[50]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Generating+text+with+recurrent+neural+networks.&btnG=
[51]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Going+deeper+with+convolutions.&btnG=
[52]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Dividethe+gradient+by+a+running+average+of+its+recent+magnitude.&btnG=
[53]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Cider:Consensus-based+image+description+evaluation.&btnG=
[54]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Show+and+tell:+A+neural+image+caption+generator.&btnG=
[55]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Aloimonos.Corpus-guided+sentence+generation+of+natural+images.&btnG=
[56]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=I2t:Image+parsing+to+text+description.&btnG=
[57]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=See+no+evil,+say+no+evil:+Description+generation+from+densely+labeled+images.&btnG=
[58]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=From+image+descriptions+to+visual+denotations:+New+similarity+metrics+for+semantic+inference+over+event+descriptions.&btnG=
[59]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Recurrent+neural+network+regularization.&btnG=
[60]: https://scholar.google.com/scholar?hl=fr&as_sdt=0%2C5&q=Learning+thevisual+interpretation+of+sentences.&btnG=

<!-- backup references in case of any error above. -->
<!-- [1] A. Barbu, A. Bridge, Z. Burchill, D. Coroian, S. Dickin-son, S. Fidler, A. Michaux, S. Mussman, S. Narayanaswamy,D. Salvi, et al. Video in sentences out. arXiv preprintarXiv:1204.2742, 2012. 2 -->
<!-- [2] K. Barnard, P. Duygulu, D. Forsyth, N. De Freitas, D. M.Blei, and M. I. Jordan. Matching words and pictures. JMLR,2003. 2 -->
<!-- [3] Y. Bengio, H. Schwenk, J.-S. Sen ́ecal, F. Morin, and J.-L.Gauvain. Neural probabilistic language models. In Innova-tions in Machine Learning. Springer, 2006. 2 -->
<!-- [4] X. Chen, H. Fang, T.-Y. Lin, R. Vedantam, S. Gupta, P. Dol-lar, and C. L. Zitnick. Microsoft coco captions: Data collec-tion and evaluation server. arXiv preprint arXiv:1504.00325,2015. 7 -->
<!-- [5] X. Chen and C. L. Zitnick. Learning a recurrent vi-sual representation for image caption generation. CoRR,abs/1411.5654, 2014. 2, 7 -->
<!-- [6] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. InCVPR, 2009. 3 -->
<!-- [7] M. Denkowski and A. Lavie. Meteor universal: Languagespecific translation evaluation for any target language. InProceedings of the EACL 2014 Workshop on Statistical Ma-chine Translation, 2014. 7 -->
<!-- [8] J. Donahue, L. A. Hendricks, S. Guadarrama, M. Rohrbach,S. Venugopalan, K. Saenko, and T. Darrell. Long-term recur-rent convolutional networks for visual recognition and de-scription. arXiv preprint arXiv:1411.4389, 2014. 2, 6, 7 -->
<!-- [9] D. Elliott and F. Keller. Image description using visual de-pendency representations. In EMNLP, pages 1292–1302,2013. 2 -->
<!-- [10] J. L. Elman. Finding structure in time. Cognitive science,14(2):179–211, 1990. 4 -->
<!-- [11] M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, andA. Zisserman. The pascal visual object classes (voc) chal-lenge. International Journal of Computer Vision, 88(2):303–338, June 2010. 1 -->
<!-- [12] H. Fang, S. Gupta, F. Iandola, R. Srivastava, L. Deng,P. Doll ́ar, J. Gao, X. He, M. Mitchell, J. Platt, et al.From captions to visual concepts and back. arXiv preprintarXiv:1411.4952, 2014. 2, 7 -->
<!-- [13] A. Farhadi, M. Hejrati, M. A. Sadeghi, P. Young,C. Rashtchian, J. Hockenmaier, and D. Forsyth. Every pic-ture tells a story: Generating sentences from images. InECCV. 2010. 1, 2 -->
<!-- [14] L. Fei-Fei, A. Iyer, C. Koch, and P. Perona. What do weperceive in a glance of a real-world scene? Journal of vision,7(1):10, 2007. 1 -->
<!-- [15] S. Fidler, A. Sharma, and R. Urtasun. A sentence is worth athousand pixels. In CVPR, 2013. 2 -->
<!-- [16] A. Frome, G. S. Corrado, J. Shlens, S. Bengio, J. Dean,T. Mikolov, et al. Devise: A deep visual-semantic embed-ding model. In NIPS, 2013. 2 -->
<!-- [17] R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich fea-ture hierarchies for accurate object detection and semanticsegmentation. In CVPR, 2014. 3 -->
<!-- [18] S. Gould, R. Fulton, and D. Koller. Decomposing a sceneinto geometric and semantically consistent regions. In Com-puter Vision, 2009 IEEE 12th International Conference on,pages 1–8. IEEE, 2009. 2 -->
<!-- [19] A. Gupta and P. Mannem. From image annotation to im-age description. In Neural information processing. Springer,2012. 2 -->
<!-- [20] S. Hochreiter and J. Schmidhuber. Long short-term memory.Neural computation, 9(8):1735–1780, 1997. 5, 7, 8 -->
<!-- [21] M. Hodosh, P. Young, and J. Hockenmaier. Framing imagedescription as a ranking task: data, models and evaluationmetrics. Journal of Artificial Intelligence Research, 2013. 1,2, 5 -->
<!-- [22] R. JeffreyPennington and C. Manning. Glove: Global vec-tors for word representation. 2014. 2 -->
<!-- [23] Y. Jia, M. Salzmann, and T. Darrell. Learning cross-modalitysimilarity for multinomial data. In ICCV, 2011. 2 -->
<!-- [24] A. Karpathy, A. Joulin, and L. Fei-Fei. Deep fragment em-beddings for bidirectional image sentence mapping. arXivpreprint arXiv:1406.5679, 2014. 2, 3, 4, 5, 6 -->
<!-- [25] R. Kiros, R. Salakhutdinov, and R. S. Zemel. Unifyingvisual-semantic embeddings with multimodal neural lan-guage models. arXiv preprint arXiv:1411.2539, 2014. 2,5, 6 -->
<!-- [26] R. Kiros, R. S. Zemel, and R. Salakhutdinov. Multimodalneural language models. ICML, 2014. 2 -->
<!-- [27] C. Kong, D. Lin, M. Basal, R. Urtasun, and S. Fidler. Whatare you talking about? text-to-image coreference. In CVPR,2014. 2 -->
<!-- [28] A. Krizhevsky, I. Sutskever, and G. E. Hinton. Imagenetclassification with deep convolutional neural networks. InNIPS, 2012. 2, 5, 7 -->
<!-- [29] G. Kulkarni, V. Premraj, S. Dhar, S. Li, Y. Choi, A. C. Berg,and T. L. Berg. Baby talk: Understanding and generatingsimple image descriptions. In CVPR, 2011. 1, 2, 3 -->
<!-- [30] P. Kuznetsova, V. Ordonez, A. C. Berg, T. L. Berg, andY. Choi. Collective generation of natural image descriptions.In ACL, 2012. 2 -->
<!-- [31] P. Kuznetsova, V. Ordonez, T. L. Berg, U. C. Hill, andY. Choi. Treetalk: Composition and compression of treesfor image descriptions. Transactions of the Association forComputational Linguistics, 2(10):351–362, 2014. 2 -->
<!-- [32] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied to document recognition. Proceed-ings of the IEEE, 86(11):2278–2324, 1998. 2 -->
<!-- [33] L.-J. Li and L. Fei-Fei. What, where and who? classifyingevents by scene and object recognition. In ICCV, 2007. 2 -->
<!-- [34] L.-J. Li, R. Socher, and L. Fei-Fei. Towards total scene un-derstanding: Classification, annotation and segmentation inan automatic framework. In Computer Vision and PatternRecognition, 2009. CVPR 2009. IEEE Conference on, pages2036–2043. IEEE, 2009. 2 -->
<!-- [35] S. Li, G. Kulkarni, T. L. Berg, A. C. Berg, and Y. Choi. Com-posing simple image descriptions using web-scale n-grams.In CoNLL, 2011. 2 -->
<!-- [36] D. Lin, S. Fidler, C. Kong, and R. Urtasun. Visual semanticsearch: Retrieving videos via complex textual queries. 2014.2n -->
<!-- [37] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ra-manan, P. Doll ́ar, and C. L. Zitnick. Microsoft coco: Com-mon objects in context. arXiv preprint arXiv:1405.0312,2014. 1, 5 -->
<!-- [38] J. Mao, W. Xu, Y. Yang, J. Wang, and A. L. Yuille. Explainimages with multimodal recurrent neural networks. arXivpreprint arXiv:1410.1090, 2014. 2, 6, 7 -->
<!-- [39] C. Matuszek*, N. FitzGerald*, L. Zettlemoyer, L. Bo, andD. Fox. A Joint Model of Language and Perception forGrounded Attribute Learning. In Proc. of the 2012 Interna-tional Conference on Machine Learning, Edinburgh, Scot-land, June 2012. 2 -->
<!-- [40] T. Mikolov, M. Karafi ́at, L. Burget, J. Cernock`y, and S. Khu-danpur. Recurrent neural network based language model. InINTERSPEECH, 2010. 2, 4 -->
<!-- [41] T. Mikolov, I. Sutskever, K. Chen, G. S. Corrado, andJ. Dean. Distributed representations of words and phrasesand their compositionality. In NIPS, 2013. 2, 3 -->
<!-- [42] M. Mitchell, X. Han, J. Dodge, A. Mensch, A. Goyal,A. Berg, K. Yamaguchi, T. Berg, K. Stratos, and H. Daum ́e,III. Midge: Generating image descriptions from computervision detections. In EACL, 2012. 2 -->
<!-- [43] V. Ordonez, G. Kulkarni, and T. L. Berg. Im2text: Describ-ing images using 1 million captioned photographs. In NIPS,2011. 2 -->
<!-- [44] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: amethod for automatic evaluation of machine translation. InProceedings of the 40th annual meeting on association forcomputational linguistics, pages 311–318. Association forComputational Linguistics, 2002. 7 -->
<!-- [45] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh,S. Ma, Z. Huang, A. Karpathy, A. Khosla, M. Bernstein,A. C. Berg, and L. Fei-Fei. Imagenet large scale visual recog-nition challenge, 2014. 1, 2, 3 -->
<!-- [46] M. Schuster and K. K. Paliwal. Bidirectional recurrent neuralnetworks. Signal Processing, IEEE Transactions on, 1997.3 -->
<!-- [47] K. Simonyan and A. Zisserman. Very deep convolutionalnetworks for large-scale image recognition. arXiv preprintarXiv:1409.1556, 2014. 5, 7 -->
<!-- [48] R. Socher and L. Fei-Fei. Connecting modalities: Semi-supervised segmentation and annotation of images using un-aligned text corpora. In CVPR, 2010. 2 -->
<!-- [49] R. Socher, A. Karpathy, Q. V. Le, C. D. Manning, and A. Y.Ng. Grounded compositional semantics for finding and de-scribing images with sentences. TACL, 2014. 2, 5, 6 -->
<!-- [50] I. Sutskever, J. Martens, and G. E. Hinton. Generating textwith recurrent neural networks. In ICML, 2011. 2, 4, 8 -->
<!-- [51] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed,D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabi-novich. Going deeper with convolutions. arXiv preprintarXiv:1409.4842, 2014. 5, 7 -->
<!-- [52] T. Tieleman and G. E. Hinton. Lecture 6.5-rmsprop: Dividethe gradient by a running average of its recent magnitude.,2012. 5 -->
<!-- [53] R. Vedantam, C. L. Zitnick, and D. Parikh. Cider:Consensus-based image description evaluation. CoRR,abs/1411.5726, 2014. 7 -->
<!-- [54] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan. Showand tell: A neural image caption generator. arXiv preprintarXiv:1411.4555, 2014. 2, 5, 6, 7 -->
<!-- [55] Y. Yang, C. L. Teo, H. Daum ́e III, and Y. Aloimonos.Corpus-guided sentence generation of natural images. InEMNLP, 2011. 2 -->
<!-- [56] B. Z. Yao, X. Yang, L. Lin, M. W. Lee, and S.-C. Zhu. I2t:Image parsing to text description. Proceedings of the IEEE,98(8):1485–1508, 2010. 2 -->
<!-- [57] M. Yatskar, L. Vanderwende, and L. Zettlemoyer. See noevil, say no evil: Description generation from densely la-beled images. Lexical and Computational Semantics, 2014.2 -->
<!-- [58] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. From im-age descriptions to visual denotations: New similarity met-rics for semantic inference over event descriptions. TACL,2014. 1, 5 -->
<!-- [59] W. Zaremba, I. Sutskever, and O. Vinyals. Recurrent neu-ral network regularization. arXiv preprint arXiv:1409.2329,2014. 5 -->
<!-- [60] C. L. Zitnick, D. Parikh, and L. Vanderwende. Learning thevisual interpretation of sentences. ICCV, 2013. 2 -->
