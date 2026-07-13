#include "Layer.h"
#include <cmath>
#include <iostream>

// ---------------------------------------------------------------------
// sigmoid(x)  ->  funzione di attivazione
// ---------------------------------------------------------------------
// La sigmoide è una funzione matematica che "schiaccia" qualsiasi
// numero reale (da -infinito a +infinito) in un valore compreso tra
// 0 e 1. La formula è:
//     sigmoide(x) = 1 / (1 + e^(-x))
//
// Perché serve? Senza una funzione di attivazione non lineare, una
// rete neurale (per quanti strati abbia) si comporterebbe come una
// semplice combinazione lineare, incapace di imparare relazioni
// complesse. La sigmoide introduce la "non linearità" necessaria e,
// avendo un output limitato tra 0 e 1, si presta bene a rappresentare
// valori normalizzati (per questo in main.cpp gli input/target vengono
// divisi per SCALE prima di essere dati alla rete).
double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

// ---------------------------------------------------------------------
// Costruttore Layer::Layer
// ---------------------------------------------------------------------
// inNodes  = numero di neuroni in ingresso a questo strato
// outNodes = numero di neuroni in uscita da questo strato
//
// La lista di inizializzazione (dopo i due punti ":") crea:
//  - weights: matrice (outNodes x inNodes). Ogni riga rappresenta un
//    neurone di uscita, ogni colonna un neurone di ingresso: quindi
//    weights[i][j] è il peso del collegamento tra l'ingresso j e
//    l'uscita i.
//  - biases: matrice colonna (outNodes x 1), un bias per ogni neurone
//    di uscita.
//
// Dopo aver creato le matrici con le dimensioni giuste, le riempiamo
// con valori casuali tra -1 e 1 chiamando randomize(): questo è il
// punto di partenza "ignorante" della rete, che verrà via via
// corretto durante l'addestramento (train, vedi NeuralNetwork.cpp).
Layer::Layer(int inNodes, int outNodes)
    : inputNodes(inNodes), outputNodes(outNodes),
      weights(outNodes, inNodes), biases(outNodes, 1)
{
    std::cout << "Costruttore Layer chiamato! "
          << inputNodes << " -> "
          << outputNodes << std::endl;

    std::cout << "Input nodes: " << inputNodes << '\n';
    std::cout << "Output nodes: " << outputNodes << '\n';

    std::cout << "Weights: "
              << weights.rows << " x "
              << weights.cols << '\n';

    std::cout << "Biases: "
              << biases.rows << " x "
              << biases.cols << '\n';

    weights.randomize();
    biases.randomize();
}

// ---------------------------------------------------------------------
// feedForward(input)  ->  calcolo "in avanti" di uno strato
// ---------------------------------------------------------------------
// Questa è l'operazione centrale di ogni strato: dato un input (una
// matrice colonna di "inputNodes" valori), calcola l'output (una
// matrice colonna di "outputNodes" valori) in tre passi:
//
//   1. output = weights * input
//      Prodotto matriciale: per ogni neurone di uscita, si calcola la
//      somma pesata di tutti gli ingressi (somma dei prodotti
//      peso * valore_input).
//
//   2. output = output + biases
//      Si aggiunge il bias di ciascun neurone di uscita. Il bias
//      permette al neurone di "attivarsi" anche quando tutti gli
//      input sono zero, dando più libertà alla rete nel modellare i
//      dati.
//
//   3. output = sigmoide(output) applicata a ogni elemento
//      Si applica la funzione di attivazione, che introduce la non
//      linearità e "normalizza" il risultato tra 0 e 1.
//
// Il risultato finale è l'output di questo strato, che diventerà
// l'input dello strato successivo (o il risultato finale della rete,
// se questo è l'ultimo strato).
Matrix Layer::feedForward(const Matrix &input)
{
    Matrix output = Matrix::multiply(weights, input);

    output.add(biases);

    output.map(sigmoid);

    return output;
}
