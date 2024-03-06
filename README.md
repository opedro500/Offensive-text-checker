## Sobre mim

Este projeto foi desenvolvido por Pedro José Cavalcanti Cabral, estudante de Tecnologia da informação - Universidade Federal do Rio Grande do Norte (UFRN).

## Projeto

O projeto conta com uma caixa de texto, onde o usuário é capaz de fazer e enviar um comentário que será avaliado pela API Perspective. Ela mostrará se o comentário é bom ou maldoso.

A API Perspective do Google usa modelos de "machine learning" para identificar comentários abusivos e linguagem "tóxica". Os modelos pontuam uma frase com base no impacto percebido que o texto pode ter em uma conversa.

## HTML

O HTML conta com um cabeçalho com informações básicas da página como o título e o link para o arquivo CSS. Já o corpo contém o container (onde possui todos os outros elementos da página), o input e o botão de submit. Além do link para o arquivo js.

## CSS

No CSS foi importado uma API do Google para estilos de fontes, além da criação de variáveis para as principais cores do projeto utlizando a pseudo-classe "root", apenas como uma forma de padronização.

## JavaScript

As seguintes operações foram feitas no arquivo js:

* Captura do valor do comentário escrito pelo usuário na variável "text".
* Chave pessoal armazenada na varável "key".
* Url do site da busca aramazenada na variável "url".
* A variável "data" foi usada para ser o corpo da requisição contendo as informações necessárias (mensagem a ser analisada, linguagem da mensagem e atributo a ser analisado) para que a API do Google possa verificar.
* A função fetch() foi utilizada para enviar uma solicitação POST para a API e receber a resposta, passando como parâmetro a url e um objeto de opções que é usado para especificar detalhes adicionais da solicitação.
* A função removeClass() foi utilizada para remover a classe "show" das div's que mostram o resultado da busca da API Perspective no documento HTML.
