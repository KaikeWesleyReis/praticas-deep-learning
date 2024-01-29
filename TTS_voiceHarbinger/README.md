# Fine-Tuning VITS (Text To Speech) para gerar a voz do Harbinger

O objetivo deste material é apresentar uma forma de fazer o fine-tuning de um modelo end to end (e2e) de TTS para geração de voz sintética para um personagem muito especial para mim:

## **Harbinger, o 1º Reaper**
<img width="1080" alt="harb" src="https://github.com/KaikeWesleyReis/praticas-deep-learning/assets/32513366/45d3859d-dca9-4e64-9b49-4f32da95a50e">

Basicamente, o código que temos aqui são dois:
- Tratamento dos áudios
- Inferência através do modelo treinado

Vale ressaltar que este material está incompleto. Falta:
- Áudios cortados manualmente (80% do trabalho está aqui)
- Transcrições realizadas através do AWS Transcribe (free tier) com revisão manual

As partes citadas acima foram as que deram mais trabalho, porém caso tenham interesse posso enviar pedidos por e-mail.
A priori, devo buscar levantar a base de dados em algum local propício para isso.

Recomendações de links que vocês precisam seguir:
- [Best Procedure For Voice Cloning - My Experience So Far](https://github.com/coqui-ai/TTS/discussions/2507)
- [Fine-tuning a 🐸 TTS model](https://docs.coqui.ai/en/latest/finetuning.html)

OBS - Não esqueçam de baixar o modelo base, é sério (perdi um mês por isso rsrsrsrs).
