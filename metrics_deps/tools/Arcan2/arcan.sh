#!/bin/bash

JARS="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
ARCAN_CLI_JAR=$(ls "${JARS}"/Arcan2-cli-*.jar)

# The path to the JVM
JAVA=java
JAVA_MEMORY="${JAVA_MEMORY:-6G}"
JVM_ARGS="--add-opens java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens java.base/sun.reflect.generics.reflectiveObjects=ALL-UNNAMED --add-opens java.base/sun.reflect.annotation=ALL-UNNAMED -Xmx${JAVA_MEMORY}"

${JAVA} ${JVM_ARGS} -cp "${JARS}/lib/*:${ARCAN_CLI_JAR}" com.arcan.Main $@ || { echo "Failed to execute Arcan."; exit 1; }