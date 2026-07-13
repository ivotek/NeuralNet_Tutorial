# Nome dell'eseguibile finale
TARGET = neural_net

# Compilatore
CXX = g++

# Flag di compilazione (per includere il supporto C++ standard e avvisi)
CXXFLAGS = -std=c++11 -Wall

# File oggetto (derivati dai file .cpp)
OBJS = Matrix.o Layer.o NeuralNetwork.o main.o

# Regola principale: come costruire l'eseguibile
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Regole per compilare i file sorgente in file oggetto
Matrix.o: Matrix.cpp Matrix.h
	$(CXX) $(CXXFLAGS) -c Matrix.cpp

Layer.o: Layer.cpp Layer.h Matrix.h
	$(CXX) $(CXXFLAGS) -c Layer.cpp

NeuralNetwork.o: NeuralNetwork.cpp NeuralNetwork.h Layer.h Matrix.h
	$(CXX) $(CXXFLAGS) -c NeuralNetwork.cpp

main.o: main.cpp NeuralNetwork.h
	$(CXX) $(CXXFLAGS) -c main.cpp

# Regola per pulire la cartella (rimuove eseguibile e file intermedi)
clean:
	rm -f $(TARGET) $(OBJS)


#
#
#	Come utilizzare il Makefile:
#    1. Assicurati di avere tutti i file (i file .cpp, i file .h e il Makefile) nella stessa cartella.
#    2. Apri il terminale (o prompt dei comandi) nella cartella del progetto.
#    3. Per compilare, scrivi semplicemente: make
#    4. Per eseguire il programma, scrivi: ./neural_net (su Windows potrebbe essere neural_net.exe).
#    5. Per pulire la cartella dai file compilati, scrivi: make clean
#Nota: Il comando make automatizza tutto. Non dovrai più compilare i file singolarmente!
