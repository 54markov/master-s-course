#include "queue.h"
#include <string.h>

queue_t msg_queue;

node_t *create_node(int id, char *value)
{
    node_t *new_node = NULL;

    new_node = (node_t *)malloc(sizeof(node_t));
    if (!new_node) {
        fprintf(stderr, "ERROR, can't allocate memory for new node\n");
        return NULL;
    }

    new_node->id       = id;
    strcpy(new_node->pay_load, value);
    new_node->next     = NULL;

    return new_node;
}

void queue_init()
{
    msg_queue.size = 0;
    msg_queue.head = NULL;
    msg_queue.tail = NULL;
}

void queue_clear()
{
    while(msg_queue.size > 0) {
        queue_dequeue();
    }

    queue_init();
}

/* push to queue */
void queue_enqueue(int id, char *new_pay_load)
{
    node_t *old_tail = msg_queue.tail;
    node_t *new_node = NULL;

    new_node = create_node(id, new_pay_load);
    if (new_node == NULL) {
        return;
    } else {
        msg_queue.tail = new_node;
    }

    if (msg_queue.head == NULL) {
        msg_queue.head = msg_queue.tail;
    } else {
        old_tail->next = msg_queue.tail;
    }

    msg_queue.size += 1;
}

/* Pop from queue */
void queue_dequeue()
{
    void *remove_value   = NULL;
    node_t *dequeue_node = NULL;
    node_t *new_head     = NULL;

    if (msg_queue.size == 0) {
        fprintf(stderr, "ERROR: can't dequeue, queue is empty\n");
        return;
    }

    /* Pop first node */
    dequeue_node = msg_queue.head;

    new_head = msg_queue.head->next;
    msg_queue.head = new_head;
    msg_queue.size -= 1;
    
    remove_value = dequeue_node->pay_load;

    free(dequeue_node);
}

void queue_print(int id)
{
    node_t *ptr = msg_queue.head;

    while(ptr) {
        if (ptr->id == id) {
            printf("%s\n", ptr->pay_load);
            ptr->id = -1;
        }
        ptr = ptr->next;
    }
}

void free_queue(int id)
{
    queue_print(id);
}

void add_to_queue(int id, char *buffer)
{
    queue_enqueue(id, buffer);
}