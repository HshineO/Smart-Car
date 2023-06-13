# -*- coding: utf-8 -*-
# !/usr/bin/python3
# utils for obs

import os
import json
import hashlib
import requests
import subprocess


class OBSUtils(object):

    def __init__(self):
        '''初始化'''
        configs = OBSUtils.get_configs()
        proxy = 'http://127.0.0.1:8003'
        self.endpoint = configs['obs_endpoint']
        self.region_name = configs['region_name']
        self.obsutil_path = '/home/ui-ide/hilens-studio/backend_modules/bin/obsutil'
        self.proxy = proxy
        self.credential = OBSUtils.get_security_token(
            configs['iam_endpoint'], proxy)
        if self.credential is None:
            print('can not get obs credential')

    def get_configs():
        resp = requests.get("http://127.0.0.1:8002/ide-configs")
        return json.loads(resp.text)

    def get_user_info():
        resp = requests.get("http://127.0.0.1:8002/user-info")
        return json.loads(resp.text)

    def get_project_token():
        resp = requests.get("http://127.0.0.1:8002/string")
        if resp.status_code < 300:
            return resp.text.replace('\n', '')
        else:
            return None

    def get_security_token(iam_endpoint_url, proxy):
        url = iam_endpoint_url + '/v3.0/OS-CREDENTIAL/securitytokens'
        token = OBSUtils.get_project_token()
        headers = {
            'Content-Type': "application/json",
            'X-Auth-Token': token
        }
        proxies = {
            'http': proxy
        }
        data = {'auth': {'identity': {'methods': ['token'], 'token': {
            'id': token, 'duration-seconds': 3600}}}}
        resp = requests.post(url, headers=headers,
                             data=json.dumps(data), proxies=proxies)
        if resp.status_code < 300:
            content = resp.json()
            credential = {}
            credential['access'] = content['credential']['access']
            credential['secret'] = content['credential']['secret']
            credential['securitytoken'] = content['credential']['securitytoken']
            return credential
        else:
            return None

    def get_default_bucket(self):
        '''获取默认的obs桶'''
        user_info = OBSUtils.get_user_info()
        if user_info['project_id'] is None:
            print('invalid project_id')
            return None

        project_id_md5 = hashlib.md5(
            user_info['project_id'].encode('utf-8')).hexdigest()[:8]
        default_bucket = 'hilens-%s-%s' % (self.region_name, project_id_md5)
        return default_bucket

    def config_obsutil(self):
        '''配置osbutil'''
        os.environ['HTTPS_PROXY'] = self.proxy
        cmd = [self.obsutil_path, 'config', '-i', self.credential['access'], '-k',
               self.credential['secret'], '-t', self.credential['securitytoken'], '-e', self.endpoint]
        ret = OBSUtils.exec_cmd_get_output(cmd, 50)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print("cannot init obsutil")
            return None

    def create_obs_bucket(self, bucket_name):
        '''创建桶'''
        os.environ['HTTPS_PROXY'] = self.proxy
        object_name = 'obs://' + bucket_name
        cmd = [self.obsutil_path, 'mb', object_name,
               '-location', self.region_name, '-e', self.endpoint]
        ret = OBSUtils.exec_cmd_get_output(cmd, 50)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print(ret)
            return None

    def put_json_data_to_obs(self, bucket_name, object_key, content):
        '''上传json数据到obs'''
        filename = 'tmp.json'
        with open(filename, 'w') as f:
            json.dump(content, f)

        object_name = 'obs://' + bucket_name + "/" + object_key
        os.environ['HTTPS_PROXY'] = self.proxy
        cmd = [self.obsutil_path, 'cp', filename,
               object_name, '-f', '-r', '-e', self.endpoint]
        ret = OBSUtils.exec_cmd_get_output(cmd, 60)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print("cannot upload data to obsfs")
            return None

    def download_obs_bucket_dir_to_local(self, bucket_name, object_key, target_dir):
        '''下载obs上的文件/文件夹到本地'''
        object_name = 'obs://' + bucket_name + "/" + object_key
        os.environ['HTTPS_PROXY'] = self.proxy
        cmd = [self.obsutil_path, 'cp', object_name,
               target_dir, '-f', '-r', '-e', self.endpoint]
        ret = OBSUtils.exec_cmd_get_output(cmd, 50)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print("cannot download data from obsfs")
            return None

    def upload_local_dir_to_obs(self, bucket_name, object_key, local_dir):
        '''上传文件/文件夹到obs'''
        object_name = 'obs://' + bucket_name + "/" + object_key
        os.environ['HTTPS_PROXY'] = self.proxy
        cmd = [self.obsutil_path, 'cp', local_dir,
               object_name, '-f', '-r', '-e', self.endpoint]
        ret = OBSUtils.exec_cmd_get_output(cmd, 60)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print("cannot upload data to obsfs")
            return None

    def sync_projects(self, bucket_name, object_key, local_dir):
        '''增量同步数据到桶中的某个文件夹'''
        object_name = 'obs://' + bucket_name + "/" + object_key
        os.environ['HTTPS_PROXY'] = self.proxy
        cmd = [self.obsutil_path, 'sync', local_dir,
               object_name, '-e', self.endpoint, '-vlength']
        ret = OBSUtils.exec_cmd_get_output(cmd, 60)
        if ret[0] == 0:
            return ret[1].strip()
        else:
            print("cannot sync data to obsfs")
            return None

    def exec_cmd_get_output(cmd, wait):
        '''执行器'''
        ret = [0, 'ok']
        try:
            retcmd = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
            if wait != 0:
                output, errout = retcmd.communicate(timeout=wait)
            else:
                output, errout = retcmd.communicate()
        except Exception as e:
            try:
                retcmd.kill()
            except Exception as ek:
                print("kill subprocess error because %s" % ek)
            print("Call linux command error because %s" % e)
            return [-1000, 'call linux command error']
        if retcmd.returncode == 0:
            ret[1] = output.decode()
        else:
            ret[0] = retcmd.returncode
            ret[1] = output.decode() + errout.decode()
        return ret
