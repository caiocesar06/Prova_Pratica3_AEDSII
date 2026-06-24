#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
  PROBLEMA DO CAIXEIRO VIAJANTE — SOLUÇÃO COMPLETA
  Algoritmos e Estruturas de Dados II
  CEFET-MG — Prof. Thiago de Souza Rodrigues — 2026
============================================================

Módulos implementados
---------------------
  Parte 1 — Força Bruta:     O((n-1)!) — todas as permutações
  Parte 2 — Heurística:      Vizinho Mais Próximo, O(n²)
  Parte 3 — Aproximado:      Baseado em MST (2-aprox), O(n²)

Uso
---
  Coloque este arquivo na MESMA pasta que si535.tsp, pa561.tsp e
  si1032.tsp, depois execute:

      python tsp_solver.py

  Saídas geradas em ./output/:
      grafico_forca_bruta.png   — gráfico pedido na Parte 1
      resultados.txt            — tabelas prontas para colar no relatório
"""

import itertools
import math
import time
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 0 — UTILITÁRIOS GERAIS
# ══════════════════════════════════════════════════════════════

def gerar_instancia(n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Gera uma instância aleatória do TSP como matriz de adjacência
    simétrica, com pesos inteiros aleatórios no intervalo [1, 1000].

    Parâmetros
    ----------
    n    : número de cidades
    seed : semente para reprodutibilidade (opcional)

    Retorno
    -------
    dist : ndarray(n, n) int64  –  dist[i][j] = custo da aresta i→j
           A diagonal principal é 0 e a matriz é simétrica.
    """
    rng = np.random.default_rng(seed)
    # Gera apenas o triângulo superior e reflete para garantir simetria
    upper = rng.integers(1, 1001, size=(n, n))
    dist = np.triu(upper, k=1)   # zera diagonal e triângulo inferior
    dist = dist + dist.T          # espelha → matriz simétrica
    return dist.astype(np.int64)


def custo_tour(dist: np.ndarray, rota: List[int]) -> int:
    """
    Calcula o custo total de um tour (ciclo hamiltoniano).

    O custo inclui a aresta de retorno: rota[-1] → rota[0].

    Parâmetros
    ----------
    dist : matriz de adjacência (n × n)
    rota : lista de n cidades sem repetição

    Retorno
    -------
    Custo inteiro total do tour.
    """
    r = np.asarray(rota, dtype=int)
    # np.roll desloca uma posição → cria pares (r[i], r[i+1 mod n])
    return int(dist[r, np.roll(r, -1)].sum())


def validar_rota(rota: List[int], n: int) -> None:
    """
    Confere que a rota é um ciclo hamiltoniano válido: visita as n
    cidades exatamente uma vez cada. Levanta AssertionError se não for.
    """
    assert len(rota) == n, f"rota tem {len(rota)} cidades, esperado {n}"
    assert len(set(rota)) == n, "rota contém cidade(s) repetida(s)"
    assert set(rota) == set(range(n)), "rota não cobre todas as cidades 0..n-1"


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 1 — FORÇA BRUTA
# ══════════════════════════════════════════════════════════════

def forca_bruta(dist: np.ndarray) -> Tuple[int, List[int]]:
    """
    Resolve o TSP por enumeração completa de todas as rotas possíveis.

    Estratégia de poda simples: fixa a cidade 0 como ponto de partida
    e permuta apenas as cidades restantes. Isso reduz o espaço de
    busca de n! para (n-1)! (elimina rotas que são rotações umas das
    outras, pois em um ciclo todos os nós são equivalentes como início).

    Complexidade
    -----------
    Tempo  : O((n-1)!)  — fatorial, cresce extremamente rápido
    Espaço : O(n)       — guarda apenas a melhor rota

    Parâmetros
    ----------
    dist : matriz de adjacência (n × n)

    Retorno
    -------
    (melhor_custo, melhor_rota)
    """
    n = dist.shape[0]

    # Casos triviais
    if n == 1:
        return 0, [0]
    if n == 2:
        return int(dist[0, 1] + dist[1, 0]), [0, 1]

    # Converte para lista Python para indexação mais rápida no loop interno
    d = dist.tolist()

    melhor_custo: float = math.inf
    melhor_rota: List[int] = []

    # Fixa cidade 0 e permuta as demais
    for perm in itertools.permutations(range(1, n)):
        rota = (0,) + perm
        # Custo do ciclo: soma das arestas consecutivas + volta ao início
        c = sum(d[rota[i]][rota[(i + 1) % n]] for i in range(n))
        if c < melhor_custo:
            melhor_custo = c
            melhor_rota = list(rota)

    return int(melhor_custo), melhor_rota


def executar_parte1(n_max: int = 16, timeout: float = 90.0) -> Dict:
    """
    Executa a Parte 1: aplica força bruta em instâncias de n = 2 até
    n_max (ou até a estimativa de tempo da PRÓXIMA instância ultrapassar
    o timeout), medindo o tempo de cada execução.

    Ajuste em relação à versão anterior
    ------------------------------------
    A versão anterior só verificava o timeout DEPOIS de rodar — o que
    significa que, se n=12 leva 1 minuto, ela ainda tentaria n=13 (que
    levaria ~12× mais, ou seja, ~12 minutos) e só então abortaria.

    Esta versão estima o tempo do PRÓXIMO n antes de rodá-lo, usando a
    relação (n-1)! = (n-2)! × (n-1), ou seja:

        tempo_estimado(n) ≈ tempo_medido(n-1) × (n-1)

    Se a estimativa já ultrapassa o timeout, a instância nem é
    iniciada. Isso é independente da velocidade da sua máquina —
    a guarda se adapta automaticamente ao hardware.

    Parâmetros
    ----------
    n_max   : tamanho máximo da instância a tentar
    timeout : teto de tempo (s) para a PRÓXIMA instância estimada

    Retorno
    -------
    Dicionário  {'n': [...], 't': [...], 'custo': [...]}
    """
    print("\n" + "═" * 64)
    print("  PARTE 1 — FORÇA BRUTA")
    print("  Instâncias aleatórias com pesos em [1, 1000]")
    print("═" * 64)
    print(f"  {'n':>4}  {'(n-1)!':>13}  {'Tempo (s)':>12}  {'Melhor custo':>13}")
    print("  " + "─" * 48)

    resultado: Dict = {'n': [], 't': [], 'custo': []}
    tempo_anterior: Optional[float] = None

    for n in range(2, n_max + 1):
        # ── Guarda preditiva: estima ANTES de rodar ─────────────────
        if tempo_anterior is not None:
            estimado = tempo_anterior * (n - 1)
            if estimado > timeout:
                print(f"  → Estimativa para n={n} é {estimado:.1f}s "
                      f"(> timeout de {timeout}s). Parando em n = {n - 1}.")
                break

        dist = gerar_instancia(n, seed=100 + n)

        t0 = time.perf_counter()
        custo, rota = forca_bruta(dist)
        dt = time.perf_counter() - t0
        validar_rota(rota, n)

        perms = math.factorial(n - 1)
        resultado['n'].append(n)
        resultado['t'].append(dt)
        resultado['custo'].append(custo)
        tempo_anterior = dt

        print(f"  {n:>4}  {perms:>13,}  {dt:>12.6f}  {custo:>13,}")

    return resultado


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 2 — HEURÍSTICA: VIZINHO MAIS PRÓXIMO (Nearest Neighbor)
# ══════════════════════════════════════════════════════════════

def vizinho_mais_proximo(dist: np.ndarray,
                          cidade_inicio: int = 0) -> Tuple[int, List[int]]:
    """
    Heurística do Vizinho Mais Próximo (Nearest Neighbor Heuristic).

    Algoritmo ganancioso (greedy):
      1. Começa em `cidade_inicio`.
      2. A cada passo, move-se à cidade não visitada mais próxima.
      3. Após visitar todas as cidades, retorna à origem.

    Vantagens
    ---------
    • Muito simples de implementar.
    • O(n²) — escala para instâncias de milhares de cidades.
    • Geralmente produz soluções ~15–25% acima do ótimo em instâncias
      euclidianas, podendo variar bastante em instâncias não-euclidianas.

    Complexidade
    -----------
    Tempo  : O(n²)  — n passos, cada um encontra o mínimo em O(n)
    Espaço : O(n)

    Parâmetros
    ----------
    dist          : matriz de adjacência (n × n)
    cidade_inicio : cidade de partida (padrão = 0)

    Retorno
    -------
    (custo_total, rota)
    """
    n = dist.shape[0]
    visitado = np.zeros(n, dtype=bool)
    rota = [cidade_inicio]
    visitado[cidade_inicio] = True

    for _ in range(n - 1):
        atual = rota[-1]
        # Copia a linha de distâncias e "bloqueia" cidades visitadas
        d_atual = dist[atual].astype(float)
        d_atual[visitado] = np.inf         # inf = ignora visitadas
        prox = int(np.argmin(d_atual))     # cidade mais próxima
        rota.append(prox)
        visitado[prox] = True

    return custo_tour(dist, rota), rota


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 3 — ALGORITMO APROXIMADO BASEADO EM MST  (2-aprox)
# ══════════════════════════════════════════════════════════════

def prim_mst(dist: np.ndarray) -> np.ndarray:
    """
    Constrói a Árvore Geradora Mínima (MST) usando o algoritmo de Prim.

    Versão otimizada para grafos densos (O(n²)) com operações
    vetorizadas do NumPy — mais eficiente que a versão com heap
    em grafos completos como o TSP.

    Algoritmo
    ---------
    Mantém key[v] = menor aresta conhecida de qualquer vértice da MST
    até v. A cada iteração, adiciona à MST o vértice com menor key.

    Parâmetros
    ----------
    dist : matriz de adjacência simétrica (n × n)

    Retorno
    -------
    parent : ndarray(n,) int  —  parent[v] = pai de v na MST
             parent[0] = -1 (raiz)
    """
    n = dist.shape[0]
    in_mst = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)          # custo mínimo para conectar v à MST
    parent = np.full(n, -1, dtype=int)
    key[0] = 0                         # começa pelo vértice 0

    for _ in range(n):
        # 1. Seleciona o vértice fora da MST com menor key
        key_masked = np.where(in_mst, np.inf, key)
        u = int(np.argmin(key_masked))
        in_mst[u] = True

        # 2. Atualiza os vizinhos de u que ainda não estão na MST
        melhora = (~in_mst) & (dist[u] < key)
        key[melhora] = dist[u, melhora]
        parent[melhora] = u

    return parent


def preorder_dfs(parent: np.ndarray, raiz: int = 0) -> List[int]:
    """
    Travessia em pré-ordem (DFS) da MST representada pelo array parent.

    Constrói primeiro a lista de filhos de cada nó a partir de parent,
    depois percorre a árvore com uma pilha iterativa (sem recursão, para
    evitar estouro de pilha de sistema em n > 1000).

    Parâmetros
    ----------
    parent : array retornado por prim_mst
    raiz   : vértice raiz (padrão = 0)

    Retorno
    -------
    Lista de vértices na ordem de visita pré-ordem da MST.
    """
    n = len(parent)
    # Constrói lista de adjacência da árvore (filhos de cada nó)
    filhos: List[List[int]] = [[] for _ in range(n)]
    for v in range(n):
        p = int(parent[v])
        if p >= 0:
            filhos[p].append(v)

    # DFS iterativa usando pilha explícita
    ordem: List[int] = []
    pilha = [raiz]
    while pilha:
        u = pilha.pop()
        ordem.append(u)
        # Empilha filhos em ordem reversa (para visitar em ordem crescente)
        for v in reversed(filhos[u]):
            pilha.append(v)

    return ordem


def aproximado_mst(dist: np.ndarray) -> Tuple[int, List[int]]:
    """
    Algoritmo 2-aproximado para o TSP métrico baseado em MST.

    Passos do algoritmo
    -------------------
    1. Constrói a MST do grafo completo.
       (custo_MST ≤ custo_OPT, pois remover uma aresta do tour ótimo
        gera uma árvore geradora)
    2. Faz a travessia em pré-ordem da MST.
       (a travessia "atalha" arestas repetidas usando a desigualdade
        triangular, sem nunca aumentar o custo)
    3. A sequência de visita forma o tour aproximado.

    Garantia teórica
    ----------------
    Para grafos métricos (desigualdade triangular):
        custo_tour ≤ 2 × custo_OPT

    Complexidade
    -----------
    Tempo  : O(n²) — dominado pelo Prim para grafos densos
    Espaço : O(n)

    Parâmetros
    ----------
    dist : matriz de adjacência simétrica (n × n)

    Retorno
    -------
    (custo_total, rota)
    """
    # Passo 1: construir MST
    parent = prim_mst(dist)
    # Passo 2: travessia pré-ordem → tour
    rota = preorder_dfs(parent, raiz=0)
    # Passo 3: custo do tour resultante
    return custo_tour(dist, rota), rota


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 4 — PARSER TSPLIB (EDGE_WEIGHT_TYPE: EXPLICIT)
# ══════════════════════════════════════════════════════════════

def parse_tsplib_explicit(filepath: str) -> Tuple[np.ndarray, int, str]:
    """
    Lê um arquivo no formato TSPLIB com EDGE_WEIGHT_TYPE: EXPLICIT e
    reconstrói a matriz de distâncias simétrica completa.

    Por que não basta ler todos os números do arquivo
    ----------------------------------------------------
    1. O arquivo tem um CABEÇALHO textual (NAME, TYPE, DIMENSION, ...)
       que não pode ser convertido para número.
    2. O formato real usado nestas instâncias é UPPER_DIAG_ROW ou
       LOWER_DIAG_ROW — ou seja, a diagonal (sempre zero) está
       INCLUÍDA nos dados. O total de valores é n(n+1)/2, não
       n(n-1)/2 como uma meia-matriz "pura" teria.
    3. Em alguns arquivos (ex.: pa561.tsp) existe uma SEGUNDA seção
       depois da matriz de distâncias (DISPLAY_DATA_SECTION, com
       coordenadas x,y de cada cidade para fins de plotagem). Esses
       números NÃO são distâncias — se forem lidos junto, a matriz
       fica corrompida.

    Esta função resolve os três pontos: localiza DIMENSION e
    EDGE_WEIGHT_FORMAT no cabeçalho, encontra o início de
    EDGE_WEIGHT_SECTION e lê exatamente n(n+1)/2 valores a partir
    daí — nem um a menos, nem um a mais.

    Parâmetros
    ----------
    filepath : caminho do arquivo .tsp

    Retorno
    -------
    (dist, n, formato)
        dist     : ndarray(n, n) int64, simétrica, diagonal zero
        n        : número de cidades (lido de DIMENSION)
        formato  : 'upper' ou 'lower' (lido de EDGE_WEIGHT_FORMAT)
    """
    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    n: Optional[int] = None
    formato: Optional[str] = None
    inicio_secao: Optional[int] = None

    for i, linha in enumerate(raw_lines):
        chave = linha.strip().upper()
        if chave.startswith('DIMENSION'):
            n = int(linha.split(':')[1].strip())
        elif chave.startswith('EDGE_WEIGHT_FORMAT'):
            valor = linha.split(':')[1].strip().upper()
            if 'UPPER' in valor:
                formato = 'upper'
            elif 'LOWER' in valor:
                formato = 'lower'
            else:
                raise ValueError(
                    f"{filepath}: EDGE_WEIGHT_FORMAT '{valor}' não suportado "
                    f"(esperado algo com UPPER ou LOWER)."
                )
        elif chave.startswith('EDGE_WEIGHT_SECTION'):
            inicio_secao = i + 1
            break  # já temos tudo que precisamos do cabeçalho

    if n is None or formato is None or inicio_secao is None:
        raise ValueError(
            f"{filepath}: cabeçalho TSPLIB incompleto — verifique se há "
            f"DIMENSION, EDGE_WEIGHT_FORMAT e EDGE_WEIGHT_SECTION."
        )

    # *_DIAG_ROW inclui a diagonal: total = n + (n-1) + ... + 1 = n(n+1)/2
    total_necessario = n * (n + 1) // 2

    # Lê tokens a partir da seção, mas PARA assim que atingir o total
    # esperado — isso ignora automaticamente qualquer seção extra
    # (como a DISPLAY_DATA_SECTION do pa561.tsp) e o marcador EOF.
    tokens: List[str] = []
    for linha in raw_lines[inicio_secao:]:
        tokens.extend(linha.split())
        if len(tokens) >= total_necessario:
            break
    tokens = tokens[:total_necessario]

    if len(tokens) != total_necessario:
        raise ValueError(
            f"{filepath}: esperados {total_necessario} valores "
            f"({formato}_diag_row, n={n}), mas encontrados {len(tokens)}. "
            f"Verifique se o arquivo não está truncado."
        )

    vals = np.array(tokens, dtype=np.int64)
    dist = np.zeros((n, n), dtype=np.int64)
    idx = 0

    if formato == 'upper':
        # UPPER_DIAG_ROW: linha i contém d[i][i], d[i][i+1], ..., d[i][n-1]
        for i in range(n):
            for j in range(i, n):
                dist[i, j] = vals[idx]
                dist[j, i] = vals[idx]
                idx += 1
    else:
        # LOWER_DIAG_ROW: linha i contém d[i][0], d[i][1], ..., d[i][i]
        for i in range(n):
            for j in range(i + 1):
                dist[i, j] = vals[idx]
                dist[j, i] = vals[idx]
                idx += 1

    return dist, n, formato


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 5 — GRÁFICO DA PARTE 1
# ══════════════════════════════════════════════════════════════

def plotar_crescimento(dados: Dict, output_path: str) -> None:
    """
    Gera e salva o gráfico de crescimento do tempo de execução da
    força bruta, mostrando o comportamento fatorial/exponencial.

    Produz dois subplots lado a lado:
      • Escala linear  — evidencia o crescimento explosivo para n grande
      • Escala log-y   — compara dados medidos com curva teórica O((n-1)!)

    Parâmetros
    ----------
    dados       : dicionário {'n': [...], 't': [...], 'custo': [...]}
    output_path : caminho onde o PNG será salvo
    """
    ns = np.array(dados['n'])
    ts = np.array(dados['t'])

    # ── Curva teórica normalizada pelo primeiro ponto com t > 0 ──────
    ref_i = next((i for i, t in enumerate(dados['t']) if t > 1e-9), 0)
    ref_n = dados['n'][ref_i]
    ref_t = dados['t'][ref_i]
    teorico = [ref_t * math.factorial(n - 1) / math.factorial(ref_n - 1)
               for n in dados['n']]

    # ── Paleta de cores ───────────────────────────────────────────────
    C_MED = '#1E40AF'    # azul escuro — dados medidos
    C_TEO = '#F59E0B'    # âmbar      — curva teórica
    C_BG = '#F8FAFC'     # cinza muito claro — fundo
    C_PT = '#64748B'     # cinza médio — anotações

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    facecolor=C_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_BG)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CBD5E1')
        ax.spines['bottom'].set_color('#CBD5E1')
        ax.tick_params(colors='#475569')

    # ── Subplot 1: escala linear ──────────────────────────────────────
    ax1.plot(ns, ts, 'o-', color=C_MED, lw=2.5, ms=8,
             mfc='white', mew=2.5, zorder=3, label='Tempo medido')
    ax1.fill_between(ns, ts, alpha=0.08, color=C_MED)
    ax1.set_xlabel('Número de cidades (n)', fontsize=12, color='#334155')
    ax1.set_ylabel('Tempo de execução (s)', fontsize=12, color='#334155')
    ax1.set_title('Escala Linear', fontsize=13, fontweight='bold',
                   color='#1E293B', pad=10)
    ax1.grid(True, alpha=0.2, linestyle='--', color='#94A3B8')
    ax1.set_xticks(ns)

    # Anota cada ponto com o tempo medido
    for x, y in zip(ns, ts):
        ax1.annotate(f'{y:.4f}s', (x, y),
                      textcoords='offset points', xytext=(0, 10),
                      ha='center', fontsize=7.5, color=C_PT)

    # ── Subplot 2: escala log ─────────────────────────────────────────
    ts_safe = np.maximum(ts, 1e-9)
    teo_safe = [max(t, 1e-12) for t in teorico]

    ax2.semilogy(ns, ts_safe, 'o-', color=C_MED, lw=2.5, ms=8,
                  mfc='white', mew=2.5, zorder=3, label='Tempo medido')
    ax2.semilogy(ns, teo_safe, '--', color=C_TEO, lw=2, alpha=0.9,
                  label='Teórico  O((n-1)!)')

    ax2.set_xlabel('Número de cidades (n)', fontsize=12, color='#334155')
    ax2.set_ylabel('Tempo de execução (s)  [escala log]',
                    fontsize=12, color='#334155')
    ax2.set_title('Escala Logarítmica', fontsize=13, fontweight='bold',
                   color='#1E293B', pad=10)
    ax2.grid(True, alpha=0.2, linestyle='--', which='both', color='#94A3B8')
    ax2.set_xticks(ns)
    ax2.legend(fontsize=10, framealpha=0.7, edgecolor='#CBD5E1')

    # ── Título geral ─────────────────────────────────────────────────
    fig.suptitle(
        'Crescimento Fatorial do Tempo  —  Força Bruta\n'
        'Problema do Caixeiro Viajante  ·  AED2 · CEFET-MG 2026',
        fontsize=14, fontweight='bold', color='#0F172A', y=1.02
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=160, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    print(f"  ✓ Gráfico salvo: {output_path}")


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 6 — EXECUÇÃO PARTES 2 E 3
# ══════════════════════════════════════════════════════════════

def executar_partes_2_3(instancias: List[Tuple[str, str]]) -> List[Dict]:
    """
    Executa a heurística (Parte 2) e o algoritmo aproximado (Parte 3)
    nas instâncias fornecidas, medindo o tempo de cada um.

    Parâmetros
    ----------
    instancias : lista de tuplas (nome, caminho_arquivo)
                 O formato (upper/lower) é detectado automaticamente
                 a partir do cabeçalho do próprio arquivo .tsp.

    Retorno
    -------
    Lista de dicionários com resultados por instância.
    """
    print("\n" + "═" * 68)
    print("  PARTE 2 — HEURÍSTICA: Vizinho Mais Próximo (NN)")
    print("  PARTE 3 — APROXIMADO: Baseado em MST (2-aprox)")
    print("═" * 68)

    resultados = []

    for nome, caminho in instancias:
        print(f"\n  ▶  Instância: {nome}")

        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Arquivo '{caminho}' não encontrado. Baixe-o do Moodle e "
                f"coloque-o na mesma pasta deste script (ou ajuste o "
                f"caminho na lista INSTANCIAS, no final do arquivo)."
            )

        dist, n, formato = parse_tsplib_explicit(caminho)
        print(f"     Lido de '{caminho}'  —  n={n} cidades, "
              f"formato detectado: {formato}_diag_row")

        # ── Parte 2: Heurística NN ───────────────────────────────────
        t0 = time.perf_counter()
        custo_nn, rota_nn = vizinho_mais_proximo(dist)
        t_nn = time.perf_counter() - t0
        validar_rota(rota_nn, n)

        # ── Parte 3: Aproximado MST ──────────────────────────────────
        t0 = time.perf_counter()
        custo_mst, rota_mst = aproximado_mst(dist)
        t_mst = time.perf_counter() - t0
        validar_rota(rota_mst, n)

        razao = custo_nn / custo_mst if custo_mst else float('inf')

        print(f"     Heurística  Vizinho Mais Próximo : {custo_nn:>14,}   ({t_nn:.3f} s)")
        print(f"     Aproximado  MST (2-aprox)        : {custo_mst:>14,}   ({t_mst:.3f} s)")
        print(f"     Razão NN / MST                   : {razao:>14.4f}")

        resultados.append({
            'nome': nome,
            'n': n,
            'custo_nn': custo_nn,
            'custo_mst': custo_mst,
            't_nn': t_nn,
            't_mst': t_mst,
            'razao': razao,
        })

    # ── Tabela sumário ───────────────────────────────────────────────
    print("\n" + "═" * 68)
    print("  SUMÁRIO — PARTES 2 & 3")
    print("═" * 68)
    print(f"  {'Instância':>10}  {'n':>6}  {'NN (Heurística)':>16}"
          f"  {'MST (Aprox)':>13}  {'Razão':>7}")
    print("  " + "─" * 59)
    for r in resultados:
        print(f"  {r['nome']:>10}  {r['n']:>6}  {r['custo_nn']:>16,}"
              f"  {r['custo_mst']:>13,}  {r['razao']:>7.4f}")

    return resultados


# ══════════════════════════════════════════════════════════════
#  SEÇÃO 7 — RELATÓRIO TEXTUAL (para colar no relatório do Moodle)
# ══════════════════════════════════════════════════════════════

def escrever_resultados_txt(dados_p1: Dict, resultados_p23: List[Dict],
                             output_path: str) -> None:
    """
    Escreve um arquivo de texto com as tabelas de resultados das três
    partes, formatadas para serem coladas diretamente no relatório.
    """
    linhas = []
    linhas.append("RESULTADOS — TRABALHO PRÁTICO TSP — AED2 — CEFET-MG")
    linhas.append("=" * 60)
    linhas.append("")
    linhas.append("PARTE 1 — Força Bruta (instâncias aleatórias)")
    linhas.append("-" * 60)
    linhas.append(f"{'n':>4} {'(n-1)!':>14} {'tempo (s)':>12} {'custo':>10}")
    for n, t, c in zip(dados_p1['n'], dados_p1['t'], dados_p1['custo']):
        linhas.append(f"{n:>4} {math.factorial(n-1):>14,} {t:>12.6f} {c:>10,}")
    linhas.append("")
    linhas.append("PARTE 2 — Heurística (Vizinho Mais Próximo)")
    linhas.append("PARTE 3 — Aproximado (MST, 2-aprox)")
    linhas.append("-" * 60)
    linhas.append(f"{'instância':>10} {'n':>6} {'NN':>14} {'MST':>14} {'razão':>8}")
    for r in resultados_p23:
        linhas.append(f"{r['nome']:>10} {r['n']:>6} {r['custo_nn']:>14,} "
                       f"{r['custo_mst']:>14,} {r['razao']:>8.4f}")
    linhas.append("")
    linhas.append("Observação: razão = custo_NN / custo_MST. Quando razão < 1,")
    linhas.append("a heurística NN encontrou um tour melhor que o aproximado")
    linhas.append("nesta instância específica — isso NÃO contradiz a garantia")
    linhas.append("teórica do MST (custo ≤ 2×OPT), que é um limite de PIOR caso,")
    linhas.append("não uma promessa de que o MST sempre vence o NN.")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(linhas))
    print(f"  ✓ Resultados em texto salvos: {output_path}")


# ══════════════════════════════════════════════════════════════
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Parte 1 ────────────────────────────────────────────────────
    # n_max=16 é só um teto de segurança; a guarda preditiva dentro de
    # executar_parte1 vai parar muito antes disso na prática (tipicamente
    # entre n=11 e n=13, dependendo da velocidade da sua máquina).
    # Para tentar ir mais longe, aumente TIMEOUT (em segundos).
    dados_p1 = executar_parte1(n_max=16, timeout=90.0)
    plotar_crescimento(dados_p1,
                        os.path.join(OUTPUT_DIR, 'grafico_forca_bruta.png'))

    # ── Partes 2 e 3 ───────────────────────────────────────────────
    # Os arquivos .tsp devem estar na mesma pasta deste script.
    # O formato (upper/lower) é detectado automaticamente do cabeçalho.
    INSTANCIAS = [
        ('si535', 'si535.tsp'),
        ('pa561', 'pa561.tsp'),
        ('si1032', 'si1032.tsp'),
    ]
    resultados_p23 = executar_partes_2_3(INSTANCIAS)

    # ── Relatório textual ───────────────────────────────────────────
    escrever_resultados_txt(dados_p1, resultados_p23,
                             os.path.join(OUTPUT_DIR, 'resultados.txt'))

    print("\n  ✓ Execução concluída. Resultados em:", OUTPUT_DIR)
