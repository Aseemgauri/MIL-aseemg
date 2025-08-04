#!/bin/bash

# Create the layout
tmux new-session -d ';' \
    split-window -v ';' \
    split-window -h ';' \
    select-pane -U ';' \
    split-window -h ';' \
    split-window -h ';' \
    select-pane -L ';' \
    select-pane -L ';' \
    split-window -h ';' \
    select-pane -R ';' \
    select-pane -R

# Move 2 left and start jack_server_start.sh
tmux select-pane -L
tmux select-pane -L
tmux send-keys './jack_server_start.sh' C-m

# Move 1 left, wait, start jack_add_sc.sh
tmux select-pane -L
tmux send-keys 'sleep 0.5 && ./jack_add_sc.sh' C-m

# Move 2 right, wait, start node_server.sh
tmux select-pane -R
tmux select-pane -R
tmux send-keys 'sleep 0.5 && ./node_server.sh' C-m

# Move down, wait, start sl_client.sh
tmux select-pane -D
tmux send-keys 'sleep 0.5 && ./sl_client.sh' C-m

# Move left, wait, start gst_client.sh
tmux select-pane -L
tmux send-keys 'sleep 0.5 && ./gst_client.sh' C-m

# Final navigation: right, up, right
tmux select-pane -R
tmux select-pane -U
tmux select-pane -R

# Attach session
tmux attach
