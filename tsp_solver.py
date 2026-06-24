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
  python tsp_solver.py               # demonstração completa
  python tsp_solver.py si535.tsp upper  # roda nas instâncias do Moodle
"""

import itertools
import math
import heapq
import time
import os
import sys
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


def executar_parte1(n_max: int = 13, timeout: float = 90.0) -> Dict:
    """
    Executa a Parte 1: aplica força bruta em instâncias de n = 2 até
    n_max (ou até estourar o timeout), medindo o tempo de cada execução.

    Parâmetros
    ----------
    n_max   : tamanho máximo da instância a tentar
    timeout : para automaticamente se uma instância levar mais de
              `timeout` segundos

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

    for n in range(2, n_max + 1):
        dist = gerar_instancia(n, seed=100 + n)

        t0 = time.perf_counter()
        custo, _ = forca_bruta(dist)
        dt = time.perf_counter() - t0

        perms = math.factorial(n - 1)
        resultado['n'].append(n)
        resultado['t'].append(dt)
        resultado['custo'].append(custo)

        print(f"  {n:>4}  {perms:>13,}  {dt:>12.6f}  {custo:>13,}")

        if dt > timeout:
            print(f"  → Tempo limite ({timeout}s) atingido. Parando em n = {n}.")
            break

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
       (cada aresta da MST é percorrida no máximo 2 vezes na travessia
        eulerianaθ; o atalho pré-ordem usa a desigualdade triangular)
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
#  SEÇÃO 4 — PARSERS PARA ARQUIVOS .TSP (MEIA-MATRIZ)
# ══════════════════════════════════════════════════════════════

def parse_half_matrix(filepath: str, triangulo: str) -> np.ndarray:
    """
    Lê um arquivo .tsp contendo uma meia-matriz de adjacência e
    reconstrói a matriz simétrica completa.

    Formato esperado
    ----------------
    O arquivo contém apenas os valores numéricos inteiros separados
    por espaços e/ou quebras de linha, sem cabeçalho.

    Diagonal SUPERIOR (si535.tsp, si1032.tsp):
        d[0][1], d[0][2], ..., d[0][n-1],
        d[1][2], d[1][3], ..., d[1][n-1],
        ...
        d[n-2][n-1]
        Total: n*(n-1)/2 valores

    Diagonal INFERIOR (pa561.tsp):
        d[1][0],
        d[2][0], d[2][1],
        ...
        d[n-1][0], d[n-1][1], ..., d[n-1][n-2]
        Total: n*(n-1)/2 valores

    Parâmetros
    ----------
    filepath  : caminho do arquivo .tsp
    triangulo : 'upper' (diagonal superior) ou 'lower' (diagonal inferior)

    Retorno
    -------
    dist : ndarray(n, n) int64 simétrica com zeros na diagonal
    """
    with open(filepath, 'r') as f:
        vals = np.array(f.read().split(), dtype=np.int64)

    total = len(vals)
    # Resolve: n*(n-1)/2 = total  ⟹  n² - n - 2*total = 0
    n = int(round((1.0 + math.sqrt(1.0 + 8.0 * total)) / 2.0))
    assert n * (n - 1) // 2 == total, (
        f"O arquivo tem {total} valores, que não corresponde a nenhuma "
        f"meia-matriz quadrada. Verifique o formato."
    )

    dist = np.zeros((n, n), dtype=np.int64)
    idx = 0

    if triangulo == 'upper':
        # Preenche triangulo superior e reflete
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = vals[idx]
                dist[j, i] = vals[idx]
                idx += 1
    else:  # 'lower'
        # Preenche triângulo inferior e reflete
        for i in range(1, n):
            for j in range(i):
                dist[i, j] = vals[idx]
                dist[j, i] = vals[idx]
                idx += 1

    return dist


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
    C_BG  = '#F8FAFC'   # cinza muito claro — fundo
    C_PT  = '#64748B'   # cinza médio — anotações

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

def executar_partes_2_3(instancias: List[Tuple[str, str, str]]) -> List[Dict]:
    """
    Executa a heurística (Parte 2) e o algoritmo aproximado (Parte 3)
    nas instâncias fornecidas, medindo o tempo de cada um.

    Parâmetros
    ----------
    instancias : lista de tuplas (nome, caminho_arquivo, triangulo)
                 triangulo = 'upper' ou 'lower'

    Retorno
    -------
    Lista de dicionários com resultados por instância.
    """
    print("\n" + "═" * 68)
    print("  PARTE 2 — HEURÍSTICA: Vizinho Mais Próximo (NN)")
    print("  PARTE 3 — APROXIMADO: Baseado em MST (2-aprox)")
    print("═" * 68)

    resultados = []

    for nome, caminho, triangulo in instancias:
        print(f"\n  ▶  Instância: {nome}  [{triangulo} triangular]")

        # Carrega a instância (arquivo real ou sintética para demo)
        if os.path.exists(caminho):
            print(f"     Lendo arquivo: {caminho}")
            dist = parse_half_matrix(caminho, triangulo)
        else:
            # Arquivo não encontrado: gera instância sintética do mesmo tamanho
            n_sintetico = {'si535': 535, 'pa561': 561,
                           'si1032': 1032}.get(nome, 200)
            print(f"     ⚠  Arquivo '{caminho}' não encontrado.")
            print(f"     → Usando instância SINTÉTICA de {n_sintetico} cidades "
                  f"(substitua pelo arquivo real do Moodle).")
            dist = gerar_instancia(n_sintetico, seed=abs(hash(nome)) % (2**31))

        n = dist.shape[0]
        print(f"     Cidades: {n:,}")

        # ── Parte 2: Heurística NN ───────────────────────────────────
        t0 = time.perf_counter()
        custo_nn, rota_nn = vizinho_mais_proximo(dist)
        t_nn = time.perf_counter() - t0

        # ── Parte 3: Aproximado MST ──────────────────────────────────
        t0 = time.perf_counter()
        custo_mst, rota_mst = aproximado_mst(dist)
        t_mst = time.perf_counter() - t0

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
#  PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Parte 1 ────────────────────────────────────────────────────
    dados_p1 = executar_parte1(n_max=13, timeout=90.0)
    plotar_crescimento(dados_p1,
                       os.path.join(OUTPUT_DIR, 'grafico_forca_bruta.png'))

    # ── Partes 2 e 3 ───────────────────────────────────────────────
    # Coloque aqui os caminhos reais dos arquivos baixados do Moodle
    INSTANCIAS = [
        ('si535',  'si535.tsp',  'upper'),
        ('pa561',  'pa561.tsp',  'lower'),
        ('si1032', 'si1032.tsp', 'upper'),
    ]
    resultados = executar_partes_2_3(INSTANCIAS)

    print("\n  ✓ Execução concluída. Resultados em:", OUTPUT_DIR)
