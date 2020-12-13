# Ingénierie logicielle pour l'IA

![Supported Python Versions](https://img.shields.io/badge/Python->=3.8-blue.svg?logo=python&logoColor=white) ![Made withJupyter](https://img.shields.io/badge/Jupyter-6.1.5-orange.svg?logo=jupyter&logoColor=white) ![Made withFlask](https://img.shields.io/badge/Flask-1.1.2-red.svg?logo=flask&logoColor=white) ![GitHub license](https://img.shields.io/badge/License-DTFW-green.svg?logo=GitHub%20Sponsors&logoColor=white)    

_Auteurs:_ [Simon Audrix](mailto:saudrix@ensc.fr) & [Gabriel Nativel-Fontaine](mailto:gnativ910e@ensc.fr)

Ce dépôt contient une API qui expose un modèle de réseau de neurones entraîné pour reconnaitre des intentions dans des phrases.

Il a été réalisé dans le cadre du module **Ingéniérie logicielle pour l'IA** du parcours **Intelligence Artificielle** inscrit dans la 3ème année du cursus d'ingénieur au sein de l'[Ecole Nationale Supérieure de Cognitique](http://www.ensc.fr).

Le dossier notebook contient les notebook nous ayant permis de répondre aux questions posées pour ce projet:

- Analyses de l'existant.ipynb répond aux questions sur l'analyse du modèle précédent (Exercices 1 & 2)
- Un nouveau modèle.ipynb répond présente notre démarche de création d'un nouveau modèle et la comparaison avec le modèle précédent (Exercice 3)
- L'API et sa documentation sont disponibles à l'addresse http://localhost:5000/ et le test de montée en charge est présenté plus bas dans ce README (Exercice 4)
- Un dockerfile est présent dans ce dépôt, l'image a également été placée sur DockerHub (Exercice 5)

## Installation

### Manuel

```shell
$ git clone https://github.com/3a-ia-ensc/glog_4_ai 
$ cd glog_4_ai 
$ docker build --tag projet-inge-log-4-ai:latest .
$ docker run --publish 5000:5000 --detach --name projet-inge-log-4-ai:latest 
```

L'API est maintenant accessible à l'adresse [http://localhost:5000/]( http://localhost:5000/)

### DockerHub

L'application est disponible sur DockerHub

```shell
$ docker pull gabrielnativelfontaine/projet-inge-log-4-ai:latest
$ docker run --publish 5000:5000 --detach --name gabrielnativelfontaine/projet-inge-log-4-ai:latest
```

L'API est maintenant accessible à l'adresse [http://localhost:5000/]( http://localhost:5000/)

## Tests de montée en charge

```shell

```

# REST API

L'API est documentée à l'adresse http://localhost:5000/

## Prédire une intention pour une phrase donnée

### Request

`GET api/intent?sentence=<<your_sentence>>`

```bash
curl -i -H 'Accept: application/json' http://localhost:5000/api/intent?sentence=<<your_sentence>>`
```

### Response

```json
HTTP/1.1 200 OK
Date: Fri, 11 Dec 2020 20:30:00 GMT
Status: 200 OK
Connection: close
Content-Type: application/json
Content-Length: 2

{
  "find-train": 0,
  "irrelevant": 0,
  "find-flight": 0,
  "find-restaurant": 0,
  "purchase": 0,
  "find-around-me": 0,
  "provide-showtimes": 0,
  "find-hotel": 0
}
```
