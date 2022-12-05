#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: ner_dataset.py
#
# Ref: https://github.com/huggingface/datasets/blob/master/datasets/conll2003/conll2003.py

import os
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import conllu
import numpy as np
from datasets import ClassLabel, DownloadConfig


class POSDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for POSDataset (UD English EWT)"""

    def __init__(self, **kwargs):
        """BuilderConfig for POSDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(POSDatasetConfig, self).__init__(**kwargs)


logger = datasets.logging.get_logger(__name__)

_CITATION = r"""\
@misc{11234/1-3424,
title = {Universal Dependencies 2.7},
author = {Zeman, Daniel and Nivre, Joakim and Abrams, Mitchell and Ackermann, Elia and Aepli, No{\"e}mi and Aghaei, Hamid and Agi{\'c}, {\v Z}eljko and Ahmadi, Amir and Ahrenberg, Lars and Ajede, Chika Kennedy and Aleksandravi{\v c}i{\=u}t{\.e}, Gabriel{\.e} and Alfina, Ika and Antonsen, Lene and Aplonova, Katya and Aquino, Angelina and Aragon, Carolina and Aranzabe, Maria Jesus and Arnard{\'o}ttir, {\t H}{\'o}runn and Arutie, Gashaw and Arwidarasti, Jessica Naraiswari and Asahara, Masayuki and Ateyah, Luma and Atmaca, Furkan and Attia, Mohammed and Atutxa, Aitziber and Augustinus, Liesbeth and Badmaeva, Elena and Balasubramani, Keerthana and Ballesteros, Miguel and Banerjee, Esha and Bank, Sebastian and Barbu Mititelu, Verginica and Basmov, Victoria and Batchelor, Colin and Bauer, John and Bedir, Seyyit Talha and Bengoetxea, Kepa and Berk, G{\"o}zde and Berzak, Yevgeni and Bhat, Irshad Ahmad and Bhat, Riyaz Ahmad and Biagetti, Erica and Bick, Eckhard and Bielinskien{\.e}, Agn{\.e} and Bjarnad{\'o}ttir, Krist{\'{\i}}n and Blokland, Rogier and Bobicev, Victoria and Boizou, Lo{\"{\i}}c and Borges V{\"o}lker, Emanuel and B{\"o}rstell, Carl and Bosco, Cristina and Bouma, Gosse and Bowman, Sam and Boyd, Adriane and Brokait{\.e}, Kristina and Burchardt, Aljoscha and Candito, Marie and Caron, Bernard and Caron, Gauthier and Cavalcanti, Tatiana and Cebiroglu Eryigit, Gulsen and Cecchini, Flavio Massimiliano and Celano, Giuseppe G. A. and Ceplo, Slavomir and Cetin, Savas and Cetinoglu, Ozlem and Chalub, Fabricio and Chi, Ethan and Cho, Yongseok and Choi, Jinho and Chun, Jayeol and Cignarella, Alessandra T. and Cinkova, Silvie and Collomb, Aurelie and Coltekin, Cagr{\i} and Connor, Miriam and Courtin, Marine and Davidson, Elizabeth and de Marneffe, Marie-Catherine and de Paiva, Valeria and Derin, Mehmet Oguz and de Souza, Elvis and Diaz de Ilarraza, Arantza and Dickerson, Carly and Dinakaramani, Arawinda and Dione, Bamba and Dirix, Peter and Dobrovoljc, Kaja and Dozat, Timothy and Droganova, Kira and Dwivedi, Puneet and Eckhoff, Hanne and Eli, Marhaba and Elkahky, Ali and Ephrem, Binyam and Erina, Olga and Erjavec, Tomaz and Etienne, Aline and Evelyn, Wograine and Facundes, Sidney and Farkas, Rich{\'a}rd and Fernanda, Mar{\'{\i}}lia and Fernandez Alcalde, Hector and Foster, Jennifer and Freitas, Cl{\'a}udia and Fujita, Kazunori and Gajdosov{\'a}, Katar{\'{\i}}na and Galbraith, Daniel and Garcia, Marcos and G{\"a}rdenfors, Moa and Garza, Sebastian and Gerardi, Fabr{\'{\i}}cio Ferraz and Gerdes, Kim and Ginter, Filip and Goenaga, Iakes and Gojenola, Koldo and G{\"o}k{\i}rmak, Memduh and Goldberg, Yoav and G{\'o}mez Guinovart, Xavier and Gonz{\'a}lez Saavedra,
Berta and Grici{\=u}t{\.e}, Bernadeta and Grioni, Matias and Grobol, Lo{\"{\i}}c and Gr{\=u}z{\={\i}}tis, Normunds and Guillaume, Bruno and Guillot-Barbance, C{\'e}line and G{\"u}ng{\"o}r, Tunga and Habash, Nizar and Hafsteinsson, Hinrik and Haji{\v c}, Jan and Haji{\v c} jr., Jan and H{\"a}m{\"a}l{\"a}inen, Mika and H{\`a} M{\~y}, Linh and Han, Na-Rae and Hanifmuti, Muhammad Yudistira and Hardwick, Sam and Harris, Kim and Haug, Dag and Heinecke, Johannes and Hellwig, Oliver and Hennig, Felix and Hladk{\'a}, Barbora and Hlav{\'a}{\v c}ov{\'a}, Jaroslava and Hociung, Florinel and Hohle, Petter and Huber, Eva and Hwang, Jena and Ikeda, Takumi and Ingason, Anton Karl and Ion, Radu and Irimia, Elena and Ishola, {\d O}l{\'a}j{\'{\i}}d{\'e} and Jel{\'{\i}}nek, Tom{\'a}{\v s} and Johannsen, Anders and J{\'o}nsd{\'o}ttir, Hildur and J{\o}rgensen, Fredrik and Juutinen, Markus and K, Sarveswaran and Ka{\c s}{\i}kara, H{\"u}ner and Kaasen, Andre and Kabaeva, Nadezhda and Kahane, Sylvain and Kanayama, Hiroshi and Kanerva, Jenna and Katz, Boris and Kayadelen, Tolga and Kenney, Jessica and Kettnerov{\'a}, V{\'a}clava and Kirchner, Jesse and Klementieva, Elena and K{\"o}hn, Arne and K{\"o}ksal, Abdullatif and Kopacewicz, Kamil and Korkiakangas, Timo and Kotsyba, Natalia and Kovalevskait{\.e}, Jolanta and Krek, Simon and Krishnamurthy, Parameswari and Kwak, Sookyoung and Laippala, Veronika and Lam, Lucia and Lambertino, Lorenzo and Lando, Tatiana and Larasati, Septina Dian and Lavrentiev, Alexei and Lee, John and L{\^e} H{\`{\^o}}ng, Phương and Lenci, Alessandro and Lertpradit, Saran and Leung, Herman and Levina, Maria and Li, Cheuk Ying and Li, Josie and Li, Keying and Li, Yuan and Lim, {KyungTae} and Linden, Krister and Ljubesic, Nikola and Loginova, Olga and Luthfi, Andry and Luukko, Mikko and Lyashevskaya, Olga and Lynn, Teresa and Macketanz, Vivien and Makazhanov, Aibek and Mandl, Michael and Manning, Christopher and Manurung, Ruli and Maranduc, Catalina and Marcek, David and Marheinecke, Katrin and Mart{\'{\i}}nez Alonso, H{\'e}ctor and Martins, Andr{\'e} and Masek, Jan and Matsuda, Hiroshi and Matsumoto, Yuji and {McDonald}, Ryan and {McGuinness}, Sarah and Mendonca, Gustavo and Miekka, Niko and Mischenkova, Karina and Misirpashayeva, Margarita and Missil{\"a}, Anna and Mititelu, Catalin and Mitrofan, Maria and Miyao, Yusuke and Mojiri Foroushani, {AmirHossein} and Moloodi, Amirsaeid and Montemagni, Simonetta and More, Amir and Moreno Romero, Laura and Mori, Keiko Sophie and Mori, Shinsuke and Morioka, Tomohiko and Moro, Shigeki and Mortensen, Bjartur and Moskalevskyi, Bohdan and Muischnek, Kadri and Munro, Robert and Murawaki, Yugo and M{\"u}{\"u}risep, Kaili and Nainwani, Pinkey and Nakhl{\'e}, Mariam and Navarro Hor{\~n}iacek, Juan Ignacio and Nedoluzhko,
Anna and Ne{\v s}pore-B{\=e}rzkalne, Gunta and Nguy{\~{\^e}}n Th{\d i}, Lương and Nguy{\~{\^e}}n Th{\d i} Minh, Huy{\`{\^e}}n and Nikaido, Yoshihiro and Nikolaev, Vitaly and Nitisaroj, Rattima and Nourian, Alireza and Nurmi, Hanna and Ojala, Stina and Ojha, Atul Kr. and Ol{\'u}{\`o}kun, Ad{\'e}day{\d o}̀ and Omura, Mai and Onwuegbuzia, Emeka and Osenova, Petya and {\"O}stling, Robert and {\O}vrelid, Lilja and {\"O}zate{\c s}, {\c S}aziye Bet{\"u}l and {\"O}zg{\"u}r, Arzucan and {\"O}zt{\"u}rk Ba{\c s}aran, Balk{\i}z and Partanen, Niko and Pascual, Elena and Passarotti, Marco and Patejuk, Agnieszka and Paulino-Passos, Guilherme and Peljak-{\L}api{\'n}ska, Angelika and Peng, Siyao and Perez, Cenel-Augusto and Perkova, Natalia and Perrier, Guy and Petrov, Slav and Petrova, Daria and Phelan, Jason and Piitulainen, Jussi and Pirinen, Tommi A and Pitler, Emily and Plank, Barbara and Poibeau, Thierry and Ponomareva, Larisa and Popel, Martin and Pretkalnina, Lauma and Pr{\'e}vost, Sophie and Prokopidis, Prokopis and Przepi{\'o}rkowski, Adam and Puolakainen, Tiina and Pyysalo, Sampo and Qi, Peng and R{\"a}{\"a}bis, Andriela and Rademaker, Alexandre and Rama, Taraka and Ramasamy, Loganathan and Ramisch, Carlos and Rashel, Fam and Rasooli, Mohammad Sadegh and Ravishankar, Vinit and Real, Livy and Rebeja, Petru and Reddy, Siva and Rehm, Georg and Riabov, Ivan and Rie{\ss}ler, Michael and Rimkut{\.e}, Erika and Rinaldi, Larissa and Rituma, Laura and Rocha, Luisa and R{\"o}gnvaldsson, Eir{\'{\i}}kur and Romanenko, Mykhailo and Rosa, Rudolf and Roșca, Valentin and Rovati, Davide and Rudina, Olga and Rueter, Jack and R{\'u}narsson, Kristjan and Sadde, Shoval and Safari, Pegah and Sagot, Benoit and Sahala, Aleksi and Saleh, Shadi and Salomoni, Alessio and Samardzi{\'c}, Tanja and Samson, Stephanie and Sanguinetti, Manuela and S{\"a}rg,
Dage and Saul{\={\i}}te, Baiba and Sawanakunanon, Yanin and Scannell, Kevin and Scarlata, Salvatore and Schneider, Nathan and Schuster, Sebastian and Seddah, Djam{\'e} and Seeker, Wolfgang and Seraji, Mojgan and Shen, Mo and Shimada, Atsuko and Shirasu, Hiroyuki and Shohibussirri, Muh and Sichinava, Dmitry and Sigurðsson, Einar Freyr and Silveira, Aline and Silveira, Natalia and Simi, Maria and Simionescu, Radu and Simk{\'o}, Katalin and {\v S}imkov{\'a}, M{\'a}ria and Simov, Kiril and Skachedubova, Maria and Smith, Aaron and Soares-Bastos, Isabela and Spadine, Carolyn and Steingr{\'{\i}}msson, Stein{\t h}{\'o}r and Stella, Antonio and Straka, Milan and Strickland, Emmett and Strnadov{\'a}, Jana and Suhr, Alane and Sulestio, Yogi Lesmana and Sulubacak, Umut and Suzuki, Shingo and Sz{\'a}nt{\'o}, Zsolt and Taji, Dima and Takahashi, Yuta and Tamburini, Fabio and Tan, Mary Ann C. and Tanaka, Takaaki and Tella, Samson and Tellier, Isabelle and Thomas, Guillaume and Torga, Liisi and Toska, Marsida and Trosterud, Trond and Trukhina, Anna and Tsarfaty, Reut and T{\"u}rk, Utku and Tyers, Francis and Uematsu, Sumire and Untilov, Roman and Uresov{\'a}, Zdenka and Uria, Larraitz and Uszkoreit, Hans and Utka, Andrius and Vajjala, Sowmya and van Niekerk, Daniel and van Noord, Gertjan and Varga, Viktor and Villemonte de la Clergerie, Eric and Vincze, Veronika and Wakasa, Aya and Wallenberg, Joel C. and Wallin, Lars and Walsh, Abigail and Wang, Jing Xian and Washington, Jonathan North and Wendt, Maximilan and Widmer, Paul and Williams, Seyi and Wir{\'e}n, Mats and Wittern, Christian and Woldemariam, Tsegay and Wong, Tak-sum and Wr{\'o}blewska, Alina and Yako, Mary and Yamashita, Kayo and Yamazaki, Naoki and Yan, Chunxiao and Yasuoka, Koichi and Yavrumyan, Marat M. and Yu, Zhuoran and Zabokrtsk{\'y}, Zdenek and Zahra, Shorouq and Zeldes, Amir and Zhu, Hanzhi and Zhuravleva, Anna},
url = {http://hdl.handle.net/11234/1-3424},
note = {{LINDAT}/{CLARIAH}-{CZ} digital library at the Institute of Formal and Applied Linguistics ({{\'U}FAL}), Faculty of Mathematics and Physics, Charles University},
copyright = {Licence Universal Dependencies v2.7},
year = {2020} }
"""
_DESCRIPTION = """\
Universal Dependencies is a project that seeks to develop cross-linguistically consistent treebank annotation for many languages, with the goal of facilitating multilingual parser development, cross-lingual learning, and parsing research from a language typology perspective. The annotation scheme is based on (universal) Stanford dependencies (de Marneffe et al., 2006, 2008, 2014), Google universal part-of-speech tags (Petrov et al., 2012), and the Interset interlingua for morphosyntactic tagsets (Zeman, 2008).
"""

CONFIGS = {"en_ewt": {"version": datasets.Version("2.11.0"),
                      "description": "UD English EWT treebank dataset, https://universaldependencies.org/treebanks/en_ewt/index.html"},
           "tweebank": {"version": datasets.Version("2.1.0"),
                        "description": "Tweebank v2 is a collection of English tweets annotated in Universal Dependencies that can be exploited for the training of NLP systems to enhance their performance on social media texts., https://github.com/Oneplus/Tweebank"}
           }

URLS = {"en_ewt": "https://share.pads.fim.uni-passau.de/datasets/nlp/en_ewt/",
        "tweebank": "https://share.pads.fim.uni-passau.de/datasets/nlp/tweebank/"}


class POSDatasetbuilder(datasets.GeneratorBasedBuilder):
    """NER dataset (conll03, ontonotes5)."""

    BUILDER_CONFIGS = []

    def __init__(self,
                 *args,
                 cache_dir,
                 dataset="en_ewt",
                 train_file="-ud-train.conllu",
                 dev_file="-ud-dev.conllu",
                 test_file="-ud-test.conllu",
                 all_file="-ud-all.conllu",
                 debugging=False,
                 **kwargs):
        self._pos_tags = self.get_labels(dataset)
        suffix_ = "_debug" if debugging else ""
        self.BUILDER_CONFIGS = [POSDatasetConfig(name=dataset + suffix_, version=CONFIGS[dataset]["version"],
                                                 description=CONFIGS[dataset]["description"])]
        self._url = URLS[dataset]
        self._train_file = f"{dataset}{train_file}"
        self._train_file_ = f"{self._train_file}.cut"
        self._dev_file = f"{dataset}{dev_file}"
        self._dev_file_ = f"{self._dev_file}.cut"
        self._test_file = f"{dataset}{test_file}"
        self._test_file_ = f"{self._test_file}.cut"
        self._all_file = f"{all_file}.cut"
        self.debugging = debugging
        self.limit = 100
        super(POSDatasetbuilder, self).__init__(*args, cache_dir=cache_dir, **kwargs)

    @classmethod
    def get_labels(cls, dataset):
        """gets the list of labels for this data set."""
        if dataset in ["en_ewt", "tweebank"]:
            return [
                "NOUN",
                "PUNCT",
                "ADP",
                "NUM",
                "SYM",
                "SCONJ",
                "ADJ",
                "PART",
                "DET",
                "CCONJ",
                "PROPN",
                "PRON",
                "X",
                "_",
                "ADV",
                "INTJ",
                "VERB",
                "AUX",
            ]
        raise ValueError(f"Dataset {dataset} is not configured. Available datasets are {CONFIGS.keys()}.")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "pos_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._pos_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="https://universaldependencies.org/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{self._url}{self._train_file}",
            "train*": f"{self._url}{self._train_file_}",
            "dev": f"{self._url}{self._dev_file}",
            "dev*": f"{self._url}{self._dev_file_}",
            "test": f"{self._url}{self._test_file}",
            "test*": f"{self._url}{self._test_file_}",
            "all*": f"{self._url}{self._all_file}",
            # "test-shuffled": f"{self._url}{self._test_file_shuffled}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="train_"),
                                    gen_kwargs={"filepath": downloaded_files["train*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="dev_"),
                                    gen_kwargs={"filepath": downloaded_files["dev*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="test_"),
                                    gen_kwargs={"filepath": downloaded_files["test*"]}),
            datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="all_"),
                                    gen_kwargs={"filepath": downloaded_files["all*"]}),
            # datasets.SplitGenerator(name=datasets.Split.__new__(datasets.Split, name="shuffled_test"),
            #                         gen_kwargs={"filepath": downloaded_files["test-shuffled"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            guid = 0
            tokenlist = list(conllu.parse_incr(f))
            for sent in tokenlist:
                if "sent_id" in sent.metadata:
                    idx = sent.metadata["sent_id"]
                else:
                    idx = guid

                yield guid, {
                    "idx": str(idx),
                    "tokens": [token["form"] for token in sent],
                    "pos_tags": [token["upos"] for token in sent],
                }
                guid += 1


class POSDataset(object):
    """
    """
    NAME = "POSDataset"

    def __init__(self, dataset="en_ewt", debugging=False):
        cache_dir = os.path.join(str(Path.home()), '.pos_datasets')
        print("Cache directory: ", cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        download_config = DownloadConfig(cache_dir=cache_dir)
        self._dataset = POSDatasetbuilder(cache_dir=cache_dir, dataset=dataset, debugging=debugging)
        print("Cache1 directory: ", self._dataset.cache_dir)
        self._dataset.download_and_prepare(download_config=download_config)
        self._dataset = self._dataset.as_dataset()
        self.sequence_lengths = self.seq_length()

    @property
    def dataset(self):
        return self._dataset

    def seq_length(self):
        train = [len(a["tokens"]) for a in self.dataset["train"]]
        dev = [len(a["tokens"]) for a in self.dataset["validation"]]
        test = [len(a["tokens"]) for a in self.dataset["test"]]
        return train + dev + test

    @property
    def labels(self) -> ClassLabel:
        return self._dataset['train'].features['ner_tags'].feature.names

    @property
    def id2label(self):
        return dict(list(enumerate(self.labels)))

    @property
    def label2id(self):
        return {v: k for k, v in self.id2label.items()}

    def train(self):
        return self._dataset['train']

    def train_(self):
        return self._dataset['train_']

    def test(self):
        return self._dataset["test"]

    def test_(self):
        return self._dataset["test_"]

    def test_shuffled(self):
        return self._dataset["shuffled_test"]

    def validation(self):
        return self._dataset["dev"]

    def validation_(self):
        return self._dataset["dev_"]

    def data(self):
        return self._dataset["all_"]


if __name__ == '__main__':
    dataset = POSDataset().dataset

    print(dataset['train'])
    print(dataset['test'])
    print(dataset['validation'])

    print("List of tags: ", dataset['train'].features['ner_tags'].feature.names)

    print("First sample: ", dataset['train'][80])

    lengths = [len(a["tokens"]) for a in dataset["train"]]
    import matplotlib

    matplotlib.use("TkAgg")
    plt.hist(lengths)
    plt.show()
    print("")
