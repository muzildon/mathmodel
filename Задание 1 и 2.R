# Esly russkie bukvy ne otobrajautsa: File -> Reopen with encoding... UTF-8

# Используйте UTF-8 как кодировку по умолчанию!
# Установить кодировку в RStudio: Tools -> Global Options -> General, 
#  Default text encoding: UTF-8

# ..............................................................................
# Математическое моделирование: Практика 1
#   Оценка точности модели с непрерывной зависимой переменной (Y)
#      * как делить данные на выборки (обучающую и тестовую)
#      * как считать MSE: среднеквадратическую ошибку модели
#      * как меняются MSE на тестовой и обучающей выборках
#        с изменением гибкости (числа степеней свободы) модели
# ..............................................................................


# Точность модели, непрерывный Y -----------------------------------------------

# Пример из лекции =============================================================

#  Генерируем данные ###########################################################

# ядро
my.seed <- 1486372882

# наблюдений всего
n.all <- 60

# доля обучающей выборки
train.percent <- 0.85
# стандартное отклонение случайного шума
res.sd <- 1
# границы изменения X
x.min <- 5
x.max <- 105

# фактические значения x
set.seed(my.seed)
x <- runif(x.min, x.max, n = n.all)
x[1:10]

# случайный шум
set.seed(my.seed)
res <- rnorm(0, res.sd, n = n.all)
res[1:10]

# отбираем наблюдения в обучающую выборку
set.seed(my.seed)
inTrain <- sample(seq_along(x), 
                  size = n.all*train.percent)
inTrain[1:10]

# истинная функция взаимосвязи 
y.func <- function(x) {	}

# для графика истинной взаимосвязи
x.line <- seq(x.min, x.max, length = n.all)
y.line <- y.func(x.line)

# фактические значения y (с шумом)
y <- y.func(x) + res
y[1:10]

# Создаём векторы с данными для построения графиков ############################

# наблюдения на обучающей выборке
x.train <- x[inTrain]

y.train <- y[inTrain]

# наблюдения на тестовой выборке
x.test <- x[-inTrain]
y.test <- y[-inTrain]

#  График 1: Исходные данные на график #########################################
png(filename = 'График 1,1.png', bg = 'transparent')

# убираем широкие поля рисунка
par(mar = c(4, 4, 1, 1))

# наименьшие/наибольшие значения по осям
x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# наблюдения с шумом (обучающая выборка)
plot(x.train, y.train, 
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# наблюдения тестовой выборки
points(x.test, y.test,
       col = 'red', bg = 'red', pch = 21)

# истинная функция
lines(x.line, y.line,
      lwd = 2, lty = 2)

# легенда
legend('topleft', legend = c('обучение', 'тест', 'f(X)'),
       pch = c(16, 16, NA), 
       col = c(grey(0.2), 'red', 'black'),  
       lty = c(0, 0, 2), lwd = c(1, 1, 2), cex = 1.2)
dev.off()

#  Строим модель ###############################################################

# модель 2 (сплайн с df = 6)
mod <- smooth.spline(x = x.train, y = y.train, df = 6)

# модельные значения для расчёта ошибок
y.model.train <- predict(mod, 
                         data.frame(x = x.train))$y[, 1]
y.model.train[1:10]
y.model.test <- predict(mod, 
                        data.frame(x = x.test))$y[, 1]
y.model.test[1:10]

# считаем средний квадрат ошибки на обучающей и тестовой выборке
MSE <- c(mean((y.train - y.model.train)^2),
         mean((y.test - y.model.test)^2))
names(MSE) <- c('train', 'test')
round(MSE, 2)

#  Цикл по степеням свободы ####################################################

# максимальное число степеней свободы для модели сплайна
max.df <- 40
# таблица для записи ошибок
tbl <- data.frame(df = 2:max.df)
# ошибки на обучающей выборке
tbl$MSE.train <- 0
# ошибки на тестовой выборке
tbl$MSE.test <- 0
head(tbl)

for (i in 2:max.df) {
    # модель
    mod <- smooth.spline(x = x.train, 
                         y = y.train, 
                         df = i)
    
    # модельные значения для расчёта ошибок
    y.model.train <- 
      predict(mod, 
              data.frame(x = x.train))$y[, 1]
    y.model.test <- 
      predict(mod, 
              data.frame(x = x.test))$y[, 1]
    
    # считаем средний квадрат ошибки на обучающей и тестовой выборке
    MSE <- c(mean((y.train - y.model.train)^2),
             mean((y.test - y.model.test)^2))
    
    # записываем результат в таблицу
    tbl[tbl$df == i, c('MSE.train', 'MSE.test')] <- MSE
}

# первые строки таблицы
head(tbl)

#  График 2: Зависимость MSE от гибкости модели ################################
png(filename = 'График 1,2,1.png', bg = 'transparent')
plot(tbl$df, tbl$MSE.test,
     type = 'l', col = 'red', lwd = 2,
     xlab = 'Степени свободы сплайна', ylab = 'MSE',
     ylim = c(min(tbl$MSE.train, tbl$MSE.test), 
              max(tbl$MSE.train, tbl$MSE.test)),
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

points(tbl$df, tbl$MSE.test,
       pch = 21, col = 'red', bg = 'red')

lines(tbl$df, tbl$MSE.train, 
      col = grey(0.3), lwd = 2)
# неустранимая ошибка
abline(h = res.sd,
       lty = 2, col = grey(0.4), lwd = 2)

# степени свободы у наименьшей ошибки на тестовой выборке
min.MSE.test <- min(tbl$MSE.test)
df.min.MSE.test <- 
  tbl[tbl$MSE.test == min.MSE.test,
      'df']

# сообщение в консоль
message(paste0('Наименьшая MSE на тестовой выборке равна ', 
               round(min.MSE.test, 2),  
               ' и достигается при df = ', df.min.MSE.test, '.'))

# компромисс между точностью и простотой модели по графику
df.my.MSE.test <- 2
my.MSE.test <- 
  tbl[tbl$df == df.my.MSE.test,
      'MSE.test']

# сообщение в консоль
message(paste0('Компромисс между точностью и сложностью модели при df = ', 
               df.my.MSE.test, ', MSE = ', round(my.MSE.test, 2), '.'))

# ставим точку на графике
abline(v = 2,
       lty = 2, lwd = 2)
points(2, my.MSE.test,
       pch = 15, col = 'blue')
mtext(df.my.MSE.test, 
      side = 1, line = -1, at = df.my.MSE.test, col = 'blue', cex = 1.2)
dev.off()
#  График 3: Лучшая модель (компромисс между гибкостью и точностью) ############
png(filename = 'График 1.3.png', bg = 'transparent')
mod.MSE.test <- 
  smooth.spline(x.train, y.train, 
                df = df.my.MSE.test)

# для гладких графиков модели
x.model.plot <- seq(x.min, x.max, length = 250)
y.model.plot <- predict(mod.MSE.test, data.frame(x = x.model.plot))$y[, 1]

# убираем широкие поля рисунка
par(mar = c(4, 4, 1, 1))

# наименьшие/наибольшие значения по осям
x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# наблюдения с шумом (обучающая выборка)
plot(x.train, y.train,
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# наблюдения тестовой выборки
points(x.test, y.test,
       col = 'red', bg = 'red', pch = 21)

# истинная функция
lines(x.line, y.line,
      lwd = 2, lty = 2)

# модель
lines(x.model.plot, y.model.plot,
      lwd = 2, col = 'blue')

# легенда
legend('topleft', legend = c('обучение', 'тест', 'f(X)', 'модель'),
       pch = c(16, 16, NA, NA), 
       col = c(grey(0.2), 'red', 'black', 'blue'),  
       lty = c(0, 0, 2, 1), lwd = c(1, 1, 2, 2), cex = 1.2)
dev.off()
#Часть 2 (изменение количества наблюдения)


# наблюдений всего
n.all <- 600
# доля обучающей выборки
train.percent <- 0.85
# стандартное отклонение случайного шума
res.sd <- 1
# границы изменения X
x.min <- 5
x.max <- 105

# фактические значения x
set.seed(my.seed)
x <- runif(x.min, x.max, n = n.all)
x[1:10]

# случайный шум
set.seed(my.seed)
res <- rnorm(0, res.sd, n = n.all)
res[1:10]

# отбираем наблюдения в обучающую выборку
set.seed(my.seed)
inTrain <- sample(seq_along(x), 
                  size = n.all*train.percent)
inTrain[1:10]

# истинная функция взаимосвязи 
y.func <- function(x) {	 8+3.5*cos(x-30)/12}

# для графика истинной взаимосвязи
x.line <- seq(x.min, x.max, length = n.all)
y.line <- y.func(x.line)

# фактические значения y (с шумом)
y <- y.func(x) + res
y[1:10]

# Создаём векторы с данными для построения графиков ############################

# наблюдения на обучающей выборке
x.train <- x[inTrain]
y.train <- y[inTrain]

# наблюдения на тестовой выборке
x.test <- x[-inTrain]
y.test <- y[-inTrain]

#  График 1: Исходные данные на график #########################################
png(filename = 'График 2,1.png', bg = 'transparent')

# убираем широкие поля рисунка
par(mar = c(4, 4, 1, 1))

# наименьшие/наибольшие значения по осям
x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# наблюдения с шумом (обучающая выборка)
plot(x.train, y.train, 
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# наблюдения тестовой выборки
points(x.test, y.test,
       col = 'red', bg = 'red', pch = 21)

# истинная функция
lines(x.line, y.line,
      lwd = 2, lty = 2)

# легенда
legend('topleft', legend = c('обучение', 'тест', 'f(X)'),
       pch = c(16, 16, NA), 
       col = c(grey(0.2), 'red', 'black'),  
       lty = c(0, 0, 2), lwd = c(1, 1, 2), cex = 1.2)
dev.off()

#  Строим модель ###############################################################

# модель 2 (сплайн с df = 6)
mod <- smooth.spline(x = x.train, y = y.train, df = 6)

# модельные значения для расчёта ошибок
y.model.train <- predict(mod, 
                         data.frame(x = x.train))$y[, 1]


y.model.train[1:10]
y.model.test <- predict(mod, 
                        data.frame(x = x.test))$y[, 1]
y.model.test[1:10]

# считаем средний квадрат ошибки на обучающей и тестовой выборке
MSE <- c(mean((y.train - y.model.train)^2),
         mean((y.test - y.model.test)^2))
names(MSE) <- c('train', 'test')
round(MSE, 2)

#  Цикл по степеням свободы ####################################################

# максимальное число степеней свободы для модели сплайна
max.df <- 40
# таблица для записи ошибок
tbl <- data.frame(df = 2:max.df)
# ошибки на обучающей выборке
tbl$MSE.train <- 0
# ошибки на тестовой выборке
tbl$MSE.test <- 0
head(tbl)

for (i in 2:max.df) {
  # модель
  mod <- smooth.spline(x = x.train, 
                       y = y.train, 
                       df = i)
  
  # модельные значения для расчёта ошибок
  y.model.train <- 
    predict(mod, 
            data.frame(x = x.train))$y[, 1]
  y.model.test <- 
    predict(mod, 
            data.frame(x = x.test))$y[, 1]
  
  # считаем средний квадрат ошибки на обучающей и тестовой выборке
  MSE <- c(mean((y.train - y.model.train)^2),
           mean((y.test - y.model.test)^2))
  
  # записываем результат в таблицу
  tbl[tbl$df == i, c('MSE.train', 'MSE.test')] <- MSE
}


# первые строки таблицы
head(tbl)

#  График 2: Зависимость MSE от гибкости модели ################################
png(filename = 'График 2,2.png', bg = 'transparent')
plot(tbl$df, tbl$MSE.test,
     type = 'l', col = 'red', lwd = 2,
     xlab = 'Степени свободы сплайна', ylab = 'MSE',
     ylim = c(min(tbl$MSE.train, tbl$MSE.test), 
              max(tbl$MSE.train, tbl$MSE.test)),
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

points(tbl$df, tbl$MSE.test,
       pch = 21, col = 'red', bg = 'red')

lines(tbl$df, tbl$MSE.train, 
      col = grey(0.3), lwd = 2)
# неустранимая ошибка
abline(h = res.sd,
       lty = 2, col = grey(0.4), lwd = 2)

# степени свободы у наименьшей ошибки на тестовой выборке
min.MSE.test <- min(tbl$MSE.test)
df.min.MSE.test <- 
  tbl[tbl$MSE.test == min.MSE.test,
      'df']

# сообщение в консоль
message(paste0('Наименьшая MSE на тестовой выборке равна ', 
               round(min.MSE.test, 2),  
               ' и достигается при df = ', df.min.MSE.test, '.'))

# компромисс между точностью и простотой модели по графику
df.my.MSE.test <- 32
my.MSE.test <- 
  tbl[tbl$df == df.my.MSE.test,
      'MSE.test']

# сообщение в консоль
message(paste0('Компромисс между точностью и сложностью модели при df = ', 
               df.my.MSE.test, ', MSE = ', round(my.MSE.test, 2), '.'))

# ставим точку на графике
abline(v = 2,
       lty = 2, lwd = 2)
points(2, my.MSE.test,
       pch = 15, col = 'blue')
mtext(df.my.MSE.test, 
      side = 1, line = -1, at = df.my.MSE.test, col = 'blue', cex = 1.2)
dev.off()
#  График 3: Лучшая модель (компромисс между гибкостью и точностью) ############
png(filename = 'График 2.3.png', bg = 'transparent')
mod.MSE.test <- 
  smooth.spline(x.train, y.train, 
                df = df.my.MSE.test)

# для гладких графиков модели
x.model.plot <- seq(x.min, x.max, length = 250)
y.model.plot <- predict(mod.MSE.test, data.frame(x = x.model.plot))$y[, 1]

# убираем широкие поля рисунка
par(mar = c(4, 4, 1, 1))

# наименьшие/наибольшие значения по осям
x.lim <- c(x.min, x.max)
y.lim <- c(min(y), max(y))

# наблюдения с шумом (обучающая выборка)
plot(x.train, y.train,
     col = grey(0.2), bg = grey(0.2), pch = 21,
     xlab = 'X', ylab = 'Y', 
     xlim = x.lim, ylim = y.lim, 
     cex = 1.2, cex.lab = 1.2, cex.axis = 1.2)

# наблюдения тестовой выборки
points(x.test, y.test,
       col = 'red', bg = 'red', pch = 21)

# истинная функция
lines(x.line, y.line,
      lwd = 2, lty = 2)

# модель
lines(x.model.plot, y.model.plot,
      lwd = 2, col = 'blue')

# легенда
legend('topleft', legend = c('обучение', 'тест', 'f(X)', 'модель'),
       pch = c(16, 16, NA, NA), 
       col = c(grey(0.2), 'red', 'black', 'blue'),  
       lty = c(0, 0, 2, 1), lwd = c(1, 1, 2, 2), cex = 1.2)
dev.off()

