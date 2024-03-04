EXECUTABLE := ./bin/exec

SRCDIR := ./src

OBJDIR := ./obj

SOURCES := $(wildcard $(SRCDIR)/*.c)

OBJECTS := $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SOURCES))

CC := gcc

CFLAGS := -std=c99 -Wall -Wextra -pedantic -ggdb
OPTFLAGS := -lm

all: $(EXECUTABLE)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(OPTFLAGS)

run:
	$(EXECUTABLE)

clean:
	rm $(OBJDIR)/*.o

purge:
	rm $(OBJDIR)/*.o $(EXECUTABLE)