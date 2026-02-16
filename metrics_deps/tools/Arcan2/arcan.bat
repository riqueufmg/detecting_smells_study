@echo off
SETLOCAL

set SPATH=%~dp0

FOR /F "tokens=*" %%g IN ('dir /b %SPATH%*.jar') do (SET ARCAN_CLI_JAR=%%g)
set ARCAN_CLI_JAR=%SPATH%\%ARCAN_CLI_JAR%

set JAVA=java
set JAVA_MEMORY=8G
set JVM_ARGS=--add-opens java.base/java.util.concurrent.atomic=ALL-UNNAMED --add-opens java.base/sun.reflect.generics.reflectiveObjects=ALL-UNNAMED --add-opens java.base/sun.reflect.annotation=ALL-UNNAMED -Xmx%JAVA_MEMORY%

%JAVA% %JVM_ARGS% -cp "%~dp0\lib\*;%ARCAN_CLI_JAR%" com.arcan.Main %*