unsigned long echo;
unsigned int distance;
unsigned long time;
int i = 0;
// unsigned long rap [5] = {0,0,0,0,0};

int deadzone = 3000;
int dtime = 50;
int lapvalue = 4;

unsigned long rap [5];
unsigned long drap [5];

void setup() {
  // put your setup code here, to run once:
  // pinMode(0,OUTPUT);
  // pinMode(1,INPUT);

  

  Serial.begin(9600);
  pinMode(1,INPUT_PULLUP);



}

void loop() {
  // put your main code here, to run repeatedly:

  

  pinMode(0,OUTPUT);
  digitalWrite(0,0);
  delayMicroseconds(2);
  digitalWrite(0,1);
  delayMicroseconds(5);
  digitalWrite(0,0);

  pinMode(0,INPUT);
  echo = pulseIn(0,1) / 2 ;
  distance = echo * 0.34442;

  if (distance < 3000 || digitalRead(1) == 0) {
    time = millis();
    drap[i] = time;
    if (i == 0){
      rap[i] = 0;
    } else {
      rap[i] = time - drap[i - 1];
    }
    i = i + 1;
    Serial.println("------------");
    Serial.print(distance);
    Serial.print(":");
    Serial.print(rap[i]);
    Serial.print("/");
    Serial.println(i);
    delay(deadzone);
  } else {
    // Serial.println(distance);
    delay(dtime);
  }

  if (i == lapvalue + 1) {
    for (int k = 0; k <= lapvalue; k++){
      Serial.print(":");
      Serial.print(rap[k]);
    }
    exit(0);  
  }


  


}
