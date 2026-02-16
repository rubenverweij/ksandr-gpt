#!/bin/bash
python3 -m ksandr.vectorstore.populate_database -env ${APP_ENV}
python3 -m ksandr.vectorstore.remove_duplicates -env ${APP_ENV}
# python3 -m ksandr.vectorstore.other_task -env ${APP_ENV}

