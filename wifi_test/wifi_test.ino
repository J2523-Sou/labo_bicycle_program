#include <WiFiS3.h>
#include <ArduinoHttpClient.h>

//wificonfig
static const char ssid[] = "LapTimer_AP";
static const char pass[] = "laptimerap";
IPAddress local_ip(192, 168, 4, 1);
IPAddress subnet(255, 255, 255, 0);

//80番ポート
WiFiServer server(80);

void setup() {

  Serial.begin(115200);
  Serial.println("WiFi beginning...");

  WiFi.config(local_ip, local_ip, subnet);

  //wifiスタートの成功確認
  if (WiFi.beginAP(ssid, pass) != WL_AP_LISTENING) {
    Serial.println("failed...");
    while (true);
  }
  Serial.print("SSID: ");
  Serial.println(ssid);
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("success");
}

void loop() {

  //リクエスト確認
  WiFiClient client = server.available();
  if (client) {
    Serial.println("PC connected");

    // HTTPリクエストを読み取る
    String request = client.readStringUntil('\r');
    Serial.println(request);

    // 簡単なHTTPレスポンスを返す
    client.println("HTTP/1.1 200 OK");
    client.println("Content-Type: text/html");
    client.println("Connection: close");
    client.println();
    client.println("<h1>Arduino Lap Timer</h1>");
    client.println("<p>connection OK</p>");
    client.stop();
  }
}
