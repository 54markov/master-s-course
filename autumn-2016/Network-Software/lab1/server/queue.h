#ifndef _QUEUE_H_
#define _QUEUE_H_

#include <stdio.h>
#include <stdlib.h>

#define BUFSIZE 1024

typedef struct node {
	int id;
    char pay_load[BUFSIZE];
    struct node  *next;
} node_t;

node_t *create_node(int id, char *value);

typedef struct {
    node_t  *head;
    node_t  *tail;
    int     size;
} queue_t;

void queue_init();                      // init queue
void queue_clear();                     // clear the queue

void queue_enqueue(int id, char *new_pay_load); // push to queue
void queue_dequeue();                           // pop from queue

int get_queue_size();                   // amount of nodes 
void queue_print(int id);

void free_queue(int id);
void add_to_queue(int id, char *buffer);

#endif