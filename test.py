import tator
token="8b2613b080cb53de709cb4665ea229236a00d66d"
host='https://cloud.tator.io'
class utils:
    
    def __init__(self,project_id,host,token):
        self.project_id=project_id
        self.api=tator.get_api(host,token)
        self.img_paths=[]

    def get_localizations_in_tmp_folder(self,localizations):
        for localization in localizations:
            
            
            img_path = self.api.get_localization_graphic(localization.id)
            print(img_path)
            print((localization))
            # print(dict(localization)["attributes"]["Fill Level"])
            self.img_paths.append(img_path)
    
    def get_localizations_list(self,media_id=None,get_localization_image=True):
        localizations=self.api.get_localization_list(85)
        if(get_localization_image):
            self.get_localizations_in_tmp_folder(localizations)
obj=utils(85,host,token)
obj.get_localizations_list()

