import pandas as pd
from ufal.udpipe import Model, Pipeline
from pymorphy2 import MorphAnalyzer
import os

UPIPE_MODEL_PATH = './russian-taiga-ud-2.3-181115.udpipe'
PATH_TO_RussianArticles_anaphora = './Data/'

model = Model.load(UPIPE_MODEL_PATH)
pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, Pipeline.DEFAULT)


class TagExtractor:
    def __init__(self, path):
        self.path = path
        self.MAX_TAGS = 100

    def _read_text(self):
        file = open(self.path)
        text = file.read()
        file.close()
        return text

    def _get_possible_first_tags(self):
        '''Генерирует список возможных тегов first в тексте'''
        possible_tags = list()
        for i in range(self.MAX_TAGS):
            possible_tags.append('<first_' + str(i) + '>')
        return possible_tags

    def find_all_index_of_substring(self, text, tag):
        start = 0
        while True:
            start = text.find(tag, start)
            if start == -1: return
            yield start
            start += len(tag)

    def _get_firsts_tags(self, text):
        df_first_tags = pd.DataFrame(columns=['tag', 'number', 'text', 'first_index', 'second_index', 'number_in_a_sentence'])
        df_first_tags['tag'] = self._get_possible_first_tags()
        for i in range(df_first_tags.shape[0]):
            indexes = list(self.find_all_index_of_substring(text, df_first_tags.tag[i]))
            if len(indexes) == 2:
                df_first_tags.number[i] = int(i)
                df_first_tags.first_index[i] = indexes[0]
                df_first_tags.second_index[i] = indexes[1]
                df_first_tags.text[i] = text[indexes[0]+len(df_first_tags.tag[i]):indexes[1]]
                df_first_tags.number_in_a_sentence[i] = len(text[:indexes[1]].split(' '))
        df_first_tags = df_first_tags.dropna()
        return df_first_tags

    def _get_possible_propn_tags(self):
        '''Генерирует список возможных тегов propn в тексте'''
        possible_tags = list()
        for i in range(self.MAX_TAGS):
            possible_tags.append('<propn_' + str(i) + '>')
        return possible_tags

    def _get_propn_tags(self, text):
        df_propn_tags = pd.DataFrame(columns=['tag', 'number', 'text', 'first_index', 'second_index', 'number_in_a_sentence'])
        df_propn_tags['tag'] = self._get_possible_propn_tags()
        for i in range(df_propn_tags.shape[0]):
            indexes = list(self.find_all_index_of_substring(text, df_propn_tags.tag[i]))
            if len(indexes) == 2:
                df_propn_tags.number[i] = int(i)
                df_propn_tags.first_index[i] = indexes[0]
                df_propn_tags.second_index[i] = indexes[1]
                df_propn_tags.text[i] = text[indexes[0]+len(df_propn_tags.tag[i]):indexes[1]]
                df_propn_tags.number_in_a_sentence[i] = len(text[:indexes[1]].split(' '))
        df_propn_tags = df_propn_tags.dropna()
        return df_propn_tags

    def _get_possible_second_tags(self):
        '''Генерирует список возможных тегов second в тексте'''
        possible_tags = list()
        for i in range(self.MAX_TAGS):
            possible_tags.append('<second_' + str(i) + '>')
        return possible_tags

    def _get_second_tags(self, text):
        df_tmp = pd.DataFrame(columns=['tag', 'number', 'text', 'first_index', 'second_index', 'number_in_a_sentence'])
        df_second_tags = pd.DataFrame(columns=['tag', 'number', 'text', 'first_index', 'second_index', 'number_in_a_sentence'])
        df_second_tags['tag'] = self._get_possible_second_tags()
        for i in range(df_second_tags.shape[0]):
            indexes = list(self.find_all_index_of_substring(text, df_second_tags.tag[i]))
            if len(indexes) == 2:
                df_second_tags.number[i] = int(i)
                df_second_tags.first_index[i] = indexes[0]
                df_second_tags.second_index[i] = indexes[1]
                df_second_tags.text[i] = text[indexes[0]+len(df_second_tags.tag[i]):indexes[1]]
                df_second_tags.number_in_a_sentence[i] = len(text[:indexes[1]].split(' '))
            elif len(indexes) >= 4 and len(indexes) % 2 == 0:
                df_second_tags.number[i] = int(i)
                df_second_tags.first_index[i] = indexes[0]
                df_second_tags.second_index[i] = indexes[1]
                df_second_tags.text[i] = text[indexes[0] + len(df_second_tags.tag[i]):indexes[1]]
                df_second_tags.number_in_a_sentence[i] = len(text[:indexes[1]].split(' '))
                for j in range(2, len(indexes), 2):
                    df_tmp = df_tmp.append({'tag': df_second_tags.tag[i],
                                            'number': int(i),
                                            'text': text[indexes[j] + len(df_second_tags.tag[i]):indexes[j+1]],
                                            'first_index': indexes[j],
                                            'second_index': indexes[j+1],
                                            'number_in_a_sentence': len(text[:indexes[j+1]].split(' '))
                                            }, ignore_index=True)
        df_second_tags = df_second_tags.dropna()
        df_second_tags = pd.concat([df_second_tags, df_tmp], ignore_index=True)
        return df_second_tags

    def extract(self):
        text = self._read_text()
        df_f = self._get_firsts_tags(text)
        df_p = self._get_propn_tags(text)
        df_s = self._get_second_tags(text)
        return df_f, df_p, df_s, text


class CombineTags:
    '''Класс для получения единой таблицы сопоставления first/propn - second'''
    def __init__(self, text, df_f, df_p, df_s):
        text = text.replace('(', '')
        text = text.replace(')', '')
        self.text = text
        self.df_f = df_f
        self.df_p = df_p
        self.df_s = df_s

    def text2pdudpipe(self, text):
        processed = pipeline.process(text)
        processed = processed.split('\n')
        data = pd.DataFrame(
            columns=['id', 'form', 'lemma', 'UPosTag', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'])
        for line in processed:
            if '#' not in line and line != '':
                data = data.append({
                    'id': line.split('\t')[0],
                    'form': line.split('\t')[1],
                    'lemma': line.split('\t')[2],
                    'UPosTag': line.split('\t')[3],
                    'XPosTag': line.split('\t')[4],
                    'Feats': line.split('\t')[5],
                    'Head': line.split('\t')[6],
                    'DepRel': line.split('\t')[7],
                    'Deps': line.split('\t')[8],
                    'Misc': line.split('\t')[9]
                }, ignore_index=True)
        # за первый проход находим все разные возможные колонки
        cols = list()
        for i in range(data.shape[0]):
            if '|' in data.Feats[i]:
                tmp_data = data.Feats[i].split('|')
                for j in tmp_data:
                    cols.append(j.split('=')[0])
        cols = list(set(cols))
        for i in range(len(cols)):
            data[cols[i]] = pd.np.nan
        for i in range(data.shape[0]):
            if '|' in data.Feats[i]:
                tmp_data = data.Feats[i].split('|')
                for j in tmp_data:
                    data[j.split('=')[0]][i] = j.split('=')[1]
        data.insert(1, 'number_of_sent', pd.np.nan)
        n = 1
        for i in range(data.shape[0] - 1):
            if data.iloc[i, 0] < data.iloc[i + 1, 0]:
                data.iloc[i, 1] = n
            else:
                data.iloc[i, 1] = n
                n += 1
        data.iloc[data.shape[0] - 1, 1] = n
        return data

    def get_udpipe_with_tag(self):
        text = self.text
        df_s = self.df_s
        df_f = self.df_f
        df_p = self.df_p

        for i in range(df_s.shape[0]):
            s1 = df_s.tag.iloc[i] + df_s.text.iloc[i] + df_s.tag.iloc[i]
            s2 = df_s.text.iloc[i] + 'S' + str(df_s.number.iloc[i])
            text = text.replace(s1, s2)

        for i in range(df_f.shape[0]):
            s1 = df_f.tag.iloc[i] + df_f.text.iloc[i] + df_f.tag.iloc[i]
            s2 = df_f.text.iloc[i] + 'F' + str(df_f.number.iloc[i])
            text = text.replace(s1, s2)

        for i in range(df_p.shape[0]):
            s1 = df_p.tag.iloc[i]
            text = text.replace(s1, '')

        return self.text2pdudpipe(text)

    def get_udpipe_without_tag(self):
        text = self.text
        df_s = self.df_s
        df_f = self.df_f
        df_p = self.df_p

        for i in range(df_s.shape[0]):
            s1 = df_s.tag.iloc[i]
            text = text.replace(s1, '')

        for i in range(df_f.shape[0]):
            s1 = df_f.tag.iloc[i]
            text = text.replace(s1, '')

        for i in range(df_p.shape[0]):
            s1 = df_p.tag.iloc[i]
            text = text.replace(s1, '')

        return self.text2pdudpipe(text)

    def get_data_about_first(self, udpipe_with_tag, udpipe_without_tag, number):

        s = 'F' + str(number)

        UPosTag, Animacy, Number, Case, Gender = pd.np.nan, pd.np.nan, pd.np.nan, pd.np.nan, pd.np.nan

        for i in range(udpipe_with_tag.shape[0]):
            if s in udpipe_with_tag.form.iloc[i]:
                UPosTag = udpipe_without_tag.UPosTag.iloc[i]
                Animacy = udpipe_without_tag.Animacy.iloc[i]
                Number = udpipe_without_tag.Number.iloc[i]
                Case = udpipe_without_tag.Case.iloc[i]
                Gender = udpipe_without_tag.Gender.iloc[i]
                break

        return UPosTag, Animacy, Number, Case, Gender

    def get_data_about_second(self, udpipe_with_tag, udpipe_without_tag, number):

        s = 'S' + str(number)

        UPosTag, Number, Person, Case, Gender = pd.np.nan, pd.np.nan, pd.np.nan, pd.np.nan, pd.np.nan

        for i in range(udpipe_with_tag.shape[0]):
            if s in udpipe_with_tag.form.iloc[i]:
                UPosTag = udpipe_without_tag.UPosTag.iloc[i]
                Person = udpipe_without_tag.Person.iloc[i]
                Number = udpipe_without_tag.Number.iloc[i]
                Case = udpipe_without_tag.Case.iloc[i]
                Gender = udpipe_without_tag.Gender.iloc[i]
                break

        return UPosTag, Number, Person, Case, Gender

    def combine(self):
        udpipe_with_tag = self.get_udpipe_with_tag()
        udpipe_without_tag = self.get_udpipe_without_tag()
        df = pd.DataFrame(columns=['UPosTag', 'Animacy', 'Number', 'Case', 'Gender', 'UPosTag.1',
                                   'Number.1', 'Person', 'Case.1', 'Gender.1', 'delta'])
        #цикл для first
        for i in range(self.df_f.shape[0]):
            number_of_related_second = self.df_s[self.df_s.number == self.df_f.number.iloc[i]].shape[0]
            if number_of_related_second >= 1:
                UPosTag, Animacy, Number, Case, Gender = self.get_data_about_first(udpipe_with_tag,
                                                                                   udpipe_without_tag,
                                                                                   self.df_f.number.iloc[i])
                for j in range(number_of_related_second):
                    UPosTag_1, Number_1, Person, Case_1, Gender_1 = self.get_data_about_second(udpipe_with_tag,
                                                                                               udpipe_without_tag,
                                                                                               self.df_s.number.iloc[j])
                    df = df.append({
                        'UPosTag': UPosTag,
                        'Animacy': Animacy,
                        'Number': Number,
                        'Case': Case,
                        'Gender': Gender,
                        'UPosTag.1': UPosTag_1,
                        'Number.1': Number_1,
                        'Person': Person,
                        'Case.1': Case_1,
                        'Gender.1': Gender_1,
                        'delta': pd.np.abs(self.df_f.number_in_a_sentence.iloc[i] - self.df_s[self.df_s.number == self.df_f.number.iloc[i]].number_in_a_sentence.iloc[j])
                    }, ignore_index=True)
        '''
        #цикл для propn
        for i in range(self.df_p.shape[0]):
            number_of_related_second = self.df_s[self.df_s.number == self.df_p.number.iloc[i]].shape[0]
            if number_of_related_second >= 1:
                for j in range(number_of_related_second):
                    df = df.append({
                        'first_or_propn': self.df_p.text.iloc[i],
                        'second': self.df_s[self.df_s.number == self.df_p.number.iloc[i]].text.iloc[j],
                        'delta': pd.np.abs(self.df_p.number_in_a_sentence.iloc[i] - self.df_s[self.df_s.number == self.df_p.number.iloc[i]].number_in_a_sentence.iloc[j])
                    }, ignore_index=True)
        '''

        return df


class FakeAssociations:
    def __init__(self, text, df_f, df_p, df_s):
        text = text.replace('(', '')
        text = text.replace(')', '')
        self.text = text
        self.df_s = df_s
        self.df_f = df_f
        self.df_p = df_p

    def convert_text(self):
        text = self.text
        df_s = self.df_s
        df_f = self.df_f
        df_p = self.df_p

        for i in range(df_s.shape[0]):
            s1 = df_s.tag.iloc[i] + df_s.text.iloc[i] + df_s.tag.iloc[i]
            s2 = df_s.text.iloc[i]+'S'+str(df_s.number.iloc[i])
            text = text.replace(s1, s2)

        for i in range(df_f.shape[0]):
            s1 = df_f.tag.iloc[i] + df_f.text.iloc[i] + df_f.tag.iloc[i]
            text = text.replace(s1, '')

        for i in range(df_p.shape[0]):
            s1 = df_p.tag.iloc[i] + df_p.text.iloc[i] + df_p.tag.iloc[i]
            text = text.replace(s1, '')

        return text

    def text2pdudpipe(self, text):
        processed = pipeline.process(text)
        processed = processed.split('\n')
        data = pd.DataFrame(
            columns=['id', 'form', 'lemma', 'UPosTag', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'])
        for line in processed:
            if '#' not in line and line != '':
                data = data.append({
                    'id': line.split('\t')[0],
                    'form': line.split('\t')[1],
                    'lemma': line.split('\t')[2],
                    'UPosTag': line.split('\t')[3],
                    'XPosTag': line.split('\t')[4],
                    'Feats': line.split('\t')[5],
                    'Head': line.split('\t')[6],
                    'DepRel': line.split('\t')[7],
                    'Deps': line.split('\t')[8],
                    'Misc': line.split('\t')[9]
                }, ignore_index=True)
        # за первый проход находим все разные возможные колонки
        cols = list()
        for i in range(data.shape[0]):
            if '|' in data.Feats[i]:
                tmp_data = data.Feats[i].split('|')
                for j in tmp_data:
                    cols.append(j.split('=')[0])
        cols = list(set(cols))
        for i in range(len(cols)):
            data[cols[i]] = pd.np.nan
        for i in range(data.shape[0]):
            if '|' in data.Feats[i]:
                tmp_data = data.Feats[i].split('|')
                for j in tmp_data:
                    data[j.split('=')[0]][i] = j.split('=')[1]
        data.insert(1, 'number_of_sent', pd.np.nan)
        n = 1
        for i in range(data.shape[0] - 1):
            if data.iloc[i, 0] < data.iloc[i + 1, 0]:
                data.iloc[i, 1] = n
            else:
                data.iloc[i, 1] = n
                n += 1
        data.iloc[data.shape[0] - 1, 1] = n
        return data

    def get_udpipe_without_tag(self):
        text = self.text
        df_s = self.df_s
        df_f = self.df_f
        df_p = self.df_p

        for i in range(df_s.shape[0]):
            s1 = df_s.tag.iloc[i] + df_s.text.iloc[i] + df_s.tag.iloc[i]
            s2 = df_s.text.iloc[i]
            text = text.replace(s1, s2)

        for i in range(df_f.shape[0]):
            s1 = df_f.tag.iloc[i] + df_f.text.iloc[i] + df_f.tag.iloc[i]
            text = text.replace(s1, '')

        for i in range(df_p.shape[0]):
            s1 = df_p.tag.iloc[i] + df_p.text.iloc[i] + df_p.tag.iloc[i]
            text = text.replace(s1, '')

        return self.text2pdudpipe(text)


    def compare_with_udpipe(self, udpipe_data, df_s):

        df_fake = pd.DataFrame(columns=['UPosTag', 'Animacy', 'Number', 'Case', 'Gender', 'UPosTag.1',
                                        'Number.1', 'Person', 'Case.1', 'Gender.1', 'delta'])

        udpipe_data_without_tag = self.get_udpipe_without_tag()

        #print(udpipe_data.shape, udpipe_data_without_tag.shape)

        #udpipe_data.to_csv('11.csv')
        #udpipe_data_without_tag.to_csv('22.csv')

        for i in range(df_s.shape[0]): #цикл по тэгам second
            s = df_s.text.iloc[i] + 'S' + str(df_s.number.iloc[i])
            for j in range(udpipe_data.shape[0]): #цикл по данным от udpipe
                if s == str(udpipe_data.form.iloc[j]):
                    if j <= 10:
                        start_point = 0
                    else:
                        start_point = j - 10
                    if udpipe_data.shape[0] - j < 10:
                        end_point = udpipe_data.shape[0]
                    else:
                        end_point = j + 10
                    for k in range(start_point, end_point): #цикл по окну в котором могут быть трудные случаи
                        if (udpipe_data.UPosTag.iloc[k] == 'NOUN') and 'F' not in udpipe_data.form.iloc[k]:
                            #print(udpipe_data.form.iloc[j], udpipe_data_without_tag.form.iloc[j])
                            df_fake = df_fake.append({
                                'UPosTag': udpipe_data.UPosTag.iloc[k],
                                'Animacy': udpipe_data.Animacy.iloc[k],
                                'Number': udpipe_data.Number.iloc[k],
                                'Case': udpipe_data.Case.iloc[k],
                                'Gender': udpipe_data.Gender.iloc[k],
                                'UPosTag.1': udpipe_data_without_tag.UPosTag.iloc[j],
                                'Number.1': udpipe_data_without_tag.Number.iloc[j],
                                'Person': udpipe_data_without_tag.Person.iloc[j],
                                'Case.1': udpipe_data_without_tag.Case.iloc[j],
                                'Gender.1': udpipe_data_without_tag.Gender.iloc[j],
                                'delta': pd.np.abs(j-k)
                            }, ignore_index=True)

        return df_fake

    def generate(self):
        text = self.convert_text()
        processed = pipeline.process(text)
        processed = processed.split('\n')
        udpipe_data = pd.DataFrame(columns=['id', 'form', 'lemma', 'UPosTag', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'])
        for line in processed:
            if '#' not in line and line != '':
                udpipe_data = udpipe_data.append({
                    'id': line.split('\t')[0],
                    'form': line.split('\t')[1],
                    'lemma': line.split('\t')[2],
                    'UPosTag': line.split('\t')[3],
                    'XPosTag': line.split('\t')[4],
                    'Feats': line.split('\t')[5],
                    'Head': line.split('\t')[6],
                    'DepRel': line.split('\t')[7],
                    'Deps': line.split('\t')[8],
                    'Misc': line.split('\t')[9]
                }, ignore_index=True)

        # за первый проход находим все разные возможные колонки
        cols = list()
        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    cols.append(j.split('=')[0])

        cols = list(set(cols))
        for i in range(len(cols)):
            udpipe_data[cols[i]] = pd.np.nan

        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    udpipe_data[j.split('=')[0]][i] = j.split('=')[1]

        udpipe_data.insert(1, 'number_of_sent', pd.np.nan)

        n = 1
        for i in range(udpipe_data.shape[0] - 1):
            if udpipe_data.iloc[i, 0] < udpipe_data.iloc[i + 1, 0]:
                udpipe_data.iloc[i, 1] = n
            else:
                udpipe_data.iloc[i, 1] = n
                n += 1
        udpipe_data.iloc[udpipe_data.shape[0] - 1, 1] = n

        df_fake = self.compare_with_udpipe(udpipe_data, self.df_s)

        return df_fake


class ConvertDataToTrain:

    def __init__(self, df):
        self.df = df

    def get_first_data(self):

        morp = MorphAnalyzer()
        udpipe_data = pd.DataFrame(
            columns=['id', 'form', 'lemma', 'UPosTag', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'])


        for text in self.df['first_or_propn'].values.tolist():

            if len(text.split(' ')) == 1:
                processed = pipeline.process(text)
            elif len(text.split(' ')) > 1:
                final_word = None
                for word in text.split(' '):
                    p = morp.parse(word)[0]
                    if str(p.tag.POS) == 'NOUN' or str(p.tag.POS) == 'PRON':
                        final_word = word
                        #processed = pipeline.process(word)
                        break
                if final_word is None:
                    processed = pipeline.process(text.split(' ')[0])
                else:
                    processed = pipeline.process(final_word)
            else:
                processed = pipeline.process(text)

            processed = processed.split('\n')
            for line in processed:
                if '#' not in line and line != '':
                    udpipe_data = udpipe_data.append({
                        'id': line.split('\t')[0],
                        'form': line.split('\t')[1],
                        'lemma': line.split('\t')[2],
                        'UPosTag': line.split('\t')[3],
                        'XPosTag': line.split('\t')[4],
                        'Feats': line.split('\t')[5],
                        'Head': line.split('\t')[6],
                        'DepRel': line.split('\t')[7],
                        'Deps': line.split('\t')[8],
                        'Misc': line.split('\t')[9]
                    }, ignore_index=True)

        # за первый проход находим все разные возможные колонки
        cols = list()
        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    cols.append(j.split('=')[0])

        cols = list(set(cols))
        for i in range(len(cols)):
            udpipe_data[cols[i]] = pd.np.nan

        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    udpipe_data[j.split('=')[0]][i] = j.split('=')[1]

        udpipe_data.insert(1, 'number_of_sent', pd.np.nan)

        n = 1
        for i in range(udpipe_data.shape[0] - 1):
            if udpipe_data.iloc[i, 0] < udpipe_data.iloc[i + 1, 0]:
                udpipe_data.iloc[i, 1] = n
            else:
                udpipe_data.iloc[i, 1] = n
                n += 1
        udpipe_data.iloc[udpipe_data.shape[0] - 1, 1] = n

        udpipe_data = udpipe_data.drop(['id', 'number_of_sent', 'form', 'lemma', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'], axis=1)

        need_cols = ['UPosTag', 'Animacy', 'Number', 'Case', 'Gender']
        for col in need_cols:
            if col not in udpipe_data:
                udpipe_data[col] = pd.np.nan

        udpipe_data = udpipe_data[['UPosTag', 'Animacy', 'Number', 'Case', 'Gender']]

        return udpipe_data

    def get_second_data(self):
        udpipe_data = pd.DataFrame(
            columns=['id', 'form', 'lemma', 'UPosTag', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'])

        for text in self.df['second'].values.tolist():

            processed = pipeline.process(text)
            processed = processed.split('\n')

            for line in processed:
                if '#' not in line and line != '':
                    udpipe_data = udpipe_data.append({
                        'id': line.split('\t')[0],
                        'form': line.split('\t')[1],
                        'lemma': line.split('\t')[2],
                        'UPosTag': line.split('\t')[3],
                        'XPosTag': line.split('\t')[4],
                        'Feats': line.split('\t')[5],
                        'Head': line.split('\t')[6],
                        'DepRel': line.split('\t')[7],
                        'Deps': line.split('\t')[8],
                        'Misc': line.split('\t')[9]
                    }, ignore_index=True)

        # за первый проход находим все разные возможные колонки
        cols = list()
        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    cols.append(j.split('=')[0])

        cols = list(set(cols))
        for i in range(len(cols)):
            udpipe_data[cols[i]] = pd.np.nan

        for i in range(udpipe_data.shape[0]):
            if '|' in udpipe_data.Feats[i]:
                tmp_data = udpipe_data.Feats[i].split('|')
                for j in tmp_data:
                    udpipe_data[j.split('=')[0]][i] = j.split('=')[1]

        udpipe_data.insert(1, 'number_of_sent', pd.np.nan)

        n = 1
        for i in range(udpipe_data.shape[0] - 1):
            if udpipe_data.iloc[i, 0] < udpipe_data.iloc[i + 1, 0]:
                udpipe_data.iloc[i, 1] = n
            else:
                udpipe_data.iloc[i, 1] = n
                n += 1
        udpipe_data.iloc[udpipe_data.shape[0] - 1, 1] = n

        udpipe_data = udpipe_data.drop(
            ['id', 'number_of_sent', 'form', 'lemma', 'XPosTag', 'Feats', 'Head', 'DepRel', 'Deps', 'Misc'], axis=1)

        need_cols = ['UPosTag', 'Number', 'Person', 'Case', 'Gender']
        for col in need_cols:
            if col not in udpipe_data:
                udpipe_data[col] = pd.np.nan
        udpipe_data = udpipe_data[['UPosTag', 'Number', 'Person', 'Case', 'Gender']]
        return udpipe_data

    def convert(self):
        data_first = self.get_first_data()
        data_second = self.get_second_data()
        data_label = self.df['label']
        data_delta = self.df['delta']
        data = pd.concat([data_first.reset_index(drop=True), data_second.reset_index(drop=True), data_label.reset_index(drop=True), data_delta.reset_index(drop=True)], axis=1)
        data.to_csv('view5.csv')
        return data


def build_data_for_train():

    data = list()

    path = PATH_TO_RussianArticles_anaphora

    files = os.listdir(path)

    for file in files:
        df_f, df_p, df_s, text = TagExtractor(path+file).extract()
        df_fake = FakeAssociations(text, df_f, df_p, df_s).generate()
        df_fake['label'] = 0
        df = CombineTags(text, df_f, df_p, df_s).combine()
        df['label'] = 1
        df_for_build_train = pd.concat([df, df_fake.sample(df.shape[0])], ignore_index=True)
        data.append(df_for_build_train)

    final_data = pd.concat(data, axis=0, ignore_index=True)
    final_data.to_csv('final.csv')


if __name__ == '__main__':
    build_data_for_train()
